use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_tungstenite::accept_async;

/// WebSocket broadcast server.
///
/// Accepts multiple clients and broadcasts state JSON to all of them.
/// Clients can connect at ws://host:port/
pub struct WsServer {
    tx: broadcast::Sender<String>,
}

impl WsServer {
    /// Start the WebSocket server on a background tokio task.
    /// Returns a WsServer handle with a `broadcast()` method.
    pub fn start(port: u16, rt: &tokio::runtime::Runtime) -> Result<Self> {
        let (tx, _) = broadcast::channel::<String>(64);
        let tx_clone = tx.clone();

        rt.spawn(async move {
            let addr = format!("0.0.0.0:{port}");
            let listener = match TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    log::error!("Failed to bind WebSocket on {addr}: {e}");
                    return;
                }
            };
            log::info!("WebSocket server listening on ws://{addr}");

            loop {
                let (stream, peer) = match listener.accept().await {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("Accept error: {e}");
                        continue;
                    }
                };

                let mut rx = tx_clone.subscribe();
                log::info!("Client connected: {peer}");

                tokio::spawn(async move {
                    let ws = match accept_async(stream).await {
                        Ok(ws) => ws,
                        Err(e) => {
                            log::warn!("WebSocket handshake failed for {peer}: {e}");
                            return;
                        }
                    };

                    let (mut writer, mut reader) = ws.split();

                    // Read task: consume client messages (pings, close frames)
                    let read_task = tokio::spawn(async move {
                        while let Some(msg) = reader.next().await {
                            match msg {
                                Ok(m) if m.is_close() => break,
                                Err(_) => break,
                                _ => {} // Ignore client messages
                            }
                        }
                    });

                    // Write task: forward broadcast messages to this client
                    let write_task = tokio::spawn(async move {
                        while let Ok(json) = rx.recv().await {
                            let msg = tokio_tungstenite::tungstenite::Message::Text(json);
                            if writer.send(msg).await.is_err() {
                                break;
                            }
                        }
                    });

                    // Wait for either task to finish
                    tokio::select! {
                        _ = read_task => {},
                        _ = write_task => {},
                    }

                    log::info!("Client disconnected: {peer}");
                });
            }
        });

        Ok(Self { tx })
    }

    /// Broadcast a JSON string to all connected clients.
    /// Non-blocking; drops the message if no clients are connected.
    pub fn broadcast(&self, json: &str) {
        let _ = self.tx.send(json.to_string());
    }

    /// Number of currently connected receivers.
    pub fn client_count(&self) -> usize {
        self.tx.receiver_count()
    }
}
