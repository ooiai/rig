//! Text Embeddings Inference (TEI) local provider
//!
//! Endpoints (router):
//! - POST /embed     -> { "inputs": string | string[] } -> { "embeddings": [[f32]] } or [ [..] ]
//! - POST /rerank    -> { "query": string, "texts": string[], "top_n"?: number } -> { "results": [{ index, text?, relevance_score }] }
//! - POST /predict   -> { "inputs": string | string[] } -> multiple shapes (items|predictions|labels+scores)
//!
//! Default base_url: http://127.0.0.1:8080
//!
//! Example curl (single input):
//! curl 127.0.0.1:8080/embed -X POST \
//!   -d '{"inputs":"What is Deep Learning?"}' \
//!   -H 'Content-Type: application/json'
use crate::client::{EmbeddingsClient, ProviderClient, VerifyClient, VerifyError};
use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use crate::impl_conversion_traits;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const TEI_DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new() -> Self {
        Self {
            base_url: TEI_DEFAULT_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        Client {
            base_url: self.base_url.to_string(),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    http_client: T,
}

impl<T> Default for Client<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Client<T>
where
    T: Default,
{
    pub fn builder<'a>() -> ClientBuilder<'a, T> {
        ClientBuilder::new()
    }

    pub fn new() -> Self {
        Self::builder().build()
    }
}

impl<T> Client<T> {
    fn req(&self, method: http_client::Method, path: &str) -> http_client::Builder {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        http_client::Builder::new().method(method).uri(url)
    }

    pub(crate) fn post(&self, path: &str) -> http_client::Builder {
        self.req(http_client::Method::POST, path)
    }
}

impl ProviderClient for Client<reqwest::Client> {
    fn from_env() -> Self {
        let base_url =
            std::env::var("TEI_BASE_URL").unwrap_or_else(|_| TEI_DEFAULT_BASE_URL.to_string());
        Self::builder().base_url(&base_url).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(base_url) = input else {
            panic!("Incorrect provider value type")
        };
        ClientBuilder::new().base_url(&base_url).build()
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        Ok(())
    }
}

impl_conversion_traits!(
    AsCompletion,
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

// ===================== Embeddings =====================
pub const TEI_TEXT_EMBEDDING: &str = "BAAI/bge-reranker-base";

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: Option<String>,
    message: Option<String>,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(
            err.error
                .or(err.message)
                .unwrap_or_else(|| "TEI error".to_string()),
        )
    }
}

#[derive(Debug, Deserialize)]
struct MultiEmbeddings {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct SingleEmbedding {
    embeddings: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingResponse {
    Multi(MultiEmbeddings),
    Single(SingleEmbedding),
    Bare(Vec<Vec<f32>>),
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Send + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();

        let inputs_value: Value = if docs.len() == 1 {
            json!({ "inputs": docs[0] })
        } else {
            json!({ "inputs": docs })
        };

        let body = serde_json::to_vec(&inputs_value)?;

        let req = self
            .client
            .post("/embed")
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = HttpClientExt::send(&self.client.http_client, req).await?;

        if !response.status().is_success() {
            let text = http_client::text(response).await?;
            return Err(EmbeddingError::ProviderError(text));
        }

        let bytes: Vec<u8> = response.into_body().await?;
        let parsed: EmbeddingResponse = serde_json::from_slice(&bytes).map_err(|e| {
            EmbeddingError::ResponseError(format!("Failed to parse TEI embeddings: {e}"))
        })?;

        let embeddings: Vec<Vec<f64>> = match parsed {
            EmbeddingResponse::Multi(m) => m
                .embeddings
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f64).collect())
                .collect(),
            EmbeddingResponse::Single(s) => {
                vec![s.embeddings.into_iter().map(|x| x as f64).collect()]
            }
            EmbeddingResponse::Bare(arr) => arr
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f64).collect())
                .collect(),
        };

        if embeddings.len() != docs.len() {
            return Err(EmbeddingError::ResponseError(
                "Response data length does not match input length".into(),
            ));
        }

        Ok(embeddings
            .into_iter()
            .zip(docs.into_iter())
            .map(|(vec, document)| embeddings::Embedding { document, vec })
            .collect())
    }
}

impl EmbeddingsClient for Client<reqwest::Client> {
    type EmbeddingModel = EmbeddingModel<reqwest::Client>;

    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, 0)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
}

// ===================== Rerank =====================

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RerankResult {
    pub index: usize,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(alias = "score", alias = "relevance_score")]
    pub relevance_score: f32,
}

#[derive(thiserror::Error, Debug)]
pub enum RerankError {
    #[error("http error: {0}")]
    Http(#[from] http_client::Error),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("response error: {0}")]
    Response(String),
}

impl Client<reqwest::Client> {
    pub async fn rerank(
        &self,
        query: &str,
        texts: impl IntoIterator<Item = String>,
        top_n: Option<usize>,
    ) -> Result<Vec<RerankResult>, RerankError> {
        let texts: Vec<String> = texts.into_iter().collect();

        let mut payload = json!({
            "query": query,
            "texts": texts,
        });
        if let Some(k) = top_n {
            payload["top_n"] = json!(k);
        }

        let body =
            serde_json::to_vec(&payload).map_err(|e| RerankError::Response(e.to_string()))?;

        let req = self
            .post("/rerank")
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| RerankError::Http(e.into()))?;

        let response = HttpClientExt::send(&self.http_client, req).await?;
        if !response.status().is_success() {
            let text = http_client::text(response).await?;
            return Err(RerankError::Provider(text));
        }

        let bytes: Vec<u8> = response.into_body().await?;
        let parsed: Vec<RerankResult> = serde_json::from_slice(&bytes).map_err(|e| {
            RerankError::Response(format!("Failed to parse TEI rerank response: {e}"))
        })?;
        Ok(parsed)
    }
}

// ===================== Predict (classification) =====================

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PredictResponse {
    pub items: Vec<LabelScore>,
}

#[derive(Debug, Deserialize)]
struct ItemsShape {
    items: Vec<LabelScore>,
}
#[derive(Debug, Deserialize)]
struct PredictionsShape {
    predictions: Vec<LabelScore>,
}
#[derive(Debug, Deserialize)]
struct ArraysShape {
    labels: Vec<String>,
    scores: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PredictResponseInternal {
    Items(ItemsShape),
    Predictions(PredictionsShape),
    Arrays(ArraysShape),
}

#[derive(thiserror::Error, Debug)]
pub enum PredictError {
    #[error("http error: {0}")]
    Http(#[from] http_client::Error),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("response error: {0}")]
    Response(String),
}

impl Client<reqwest::Client> {
    /// Predict/classify inputs using TEI router /predict
    pub async fn predict(
        &self,
        inputs: impl IntoIterator<Item = String>,
    ) -> Result<PredictResponse, PredictError> {
        let inputs_vec: Vec<String> = inputs.into_iter().collect();
        let body_value = if inputs_vec.len() == 1 {
            json!({ "inputs": inputs_vec[0] })
        } else {
            json!({ "inputs": inputs_vec })
        };

        let body =
            serde_json::to_vec(&body_value).map_err(|e| PredictError::Response(e.to_string()))?;

        let req = self
            .post("/predict")
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| PredictError::Http(e.into()))?;

        let response = HttpClientExt::send(&self.http_client, req).await?;
        if !response.status().is_success() {
            let text = http_client::text(response).await?;
            return Err(PredictError::Provider(text));
        }

        let bytes: Vec<u8> = response.into_body().await?;
        let internal: PredictResponseInternal = serde_json::from_slice(&bytes).map_err(|e| {
            PredictError::Response(format!("Failed to parse TEI predict response: {e}"))
        })?;

        let items = match internal {
            PredictResponseInternal::Items(x) => x.items,
            PredictResponseInternal::Predictions(x) => x.predictions,
            PredictResponseInternal::Arrays(x) => {
                if x.labels.len() != x.scores.len() {
                    return Err(PredictError::Response(
                        "labels and scores length mismatch".into(),
                    ));
                }
                x.labels
                    .into_iter()
                    .zip(x.scores.into_iter())
                    .map(|(label, score)| LabelScore { label, score })
                    .collect()
            }
        };

        Ok(PredictResponse { items })
    }
}
