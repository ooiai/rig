use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::volcengine;

#[tokio::main]
async fn main() {
    // Create Volcengine client and model
    let client = volcengine::Client::from_env();
    // let client = volcengine::Client::builder(&key.api_key)
    //     .base_url(&key.endpoint)
    //     .build();
    // let context = BASE64_STANDARD.encode("xxxxx");
    let agent = client
        .agent("ep-20250211190211-hlpsc")
        // .context("I'm boy")
        // .context("I'm girl")
        .build();

    // Prompt the model and print its response
    let response = agent
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt Volcengine");

    println!("Volcengine: {response}");
}
