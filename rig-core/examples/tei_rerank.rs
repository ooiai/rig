use rig::providers::tei;
use std::env;

fn tei_base_url() -> String {
    env::var("TEI_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:6290".to_string())
}

fn make_client() -> tei::Client {
    tei::Client::builder().base_url(&tei_base_url()).build()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = make_client();

    let query = "deep learning";
    let texts = vec![
        "Deep learning uses neural networks.".to_string(),
        "Apples and oranges.".to_string(),
        "Backpropagation algorithm.".to_string(),
    ];

    match client.rerank(query, texts.clone(), Some(2)).await {
        Ok(r) => {
            println!("Rerank: got {} results (top_n=2)", r.len());
            for (i, item) in r.iter().enumerate() {
                println!(
                    "[rerank] #{} index={} score={:.3} text={:?}",
                    i, item.index, item.relevance_score, item.text
                );
            }
        }
        Err(err) => {
            eprintln!(
                "Rerank failed: {err:?}. Ensure TEI router supports /rerank at {}",
                tei_base_url()
            );
        }
    }

    Ok(())
}
