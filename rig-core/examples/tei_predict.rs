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

    let inputs = vec!["This library is amazing!".to_string()];
    match client.predict(inputs.clone()).await {
        Ok(out) => {
            println!("Predict: {} items", out.items.len());
            for (i, it) in out.items.iter().enumerate() {
                println!(
                    "[predict] #{} label='{}' score={:.3}",
                    i, it.label, it.score
                );
            }
        }
        Err(err) => {
            eprintln!(
                "Predict failed: {err:?}. Ensure TEI router supports /predict at {}",
                tei_base_url()
            );
        }
    }

    Ok(())
}
