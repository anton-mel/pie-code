//! A check of pie's `session.send-to`, the pipe between processes. Launched
//! with no peer, it reports its id and echoes whatever reaches its inbox.
//! Launched with a peer, it sends the peer `text` through the engine and
//! reports whether the send went through.

use inferlet::{runtime, session};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    peer: String,
    #[serde(default)]
    text: String,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String, String> {
    if input.peer.is_empty() {
        session::send(&format!(r#"{{"id":"{}"}}"#, runtime::instance_id()));
        while let Some(message) = session::receive().await {
            if message == "quit" {
                break;
            }
            session::send(&format!("got: {message}"));
        }
        return Ok(String::new());
    }
    let sent = session::send_to(&input.peer, &input.text);
    session::send(&format!("sent: {sent:?}"));
    Ok(String::new())
}
