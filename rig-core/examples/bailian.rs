use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::bailian;

#[tokio::main]
async fn main() {
    let client = bailian::Client::from_env();
    let agent = client
        .agent("qwen3-max")
        // .context("I'm boy")
        // .context("I'm girl")
        .build();

    // Prompt the model and print its response
    let response = agent
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt Bailian");

    println!("Bailian: {response}");
}
