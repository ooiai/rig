use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModelDyn;
use rig::providers::tei;
use std::env;

fn tei_base_url() -> String {
    env::var("TEI_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:6280".to_string())
}

fn make_client() -> tei::Client {
    tei::Client::builder().base_url(&tei_base_url()).build()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = make_client();

    // 1) Embedding
    let model = client.embedding_model("local-tei");
    let docs = vec!["What is Deep Learning?".to_string(), "Hello!".to_string()];

    let embs = model.embed_texts(docs.clone()).await?;
    println!("Embeddings: count = {}", embs.len());
    for (i, e) in embs.iter().take(2).enumerate() {
        println!("[embed] #{} dim={} doc='{}'", i, e.vec.len(), e.document);
    }

    Ok(())
}
