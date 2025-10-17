use rig::Embed;
use rig::prelude::*;
use rig::providers::volcengine;

#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Volcengine client and model
    let client = volcengine::Client::from_env();

    // Prompt the model and print its response
    let embeddings = client
        .embeddings("doubao-embedding-text-240715")
        .document(Greetings {
            message: "Hello, world!".to_string(),
        })?
        .document(Greetings {
            message: "Goodbye, world!".to_string(),
        })?
        .build()
        .await
        .expect("Failed to embed documents");

    println!("{embeddings:?}");
    Ok(())
}
