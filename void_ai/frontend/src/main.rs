use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use gloo_net::eventsource::futures::EventSource;
#[allow(unused_imports)]
use futures::StreamExt;
use gloo_net::http::Request;
use wasm_bindgen::JsCast; // Added this import!

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FileNode {
    name: String,
    is_dir: bool,
    children: Option<Vec<FileNode>>,
}

#[component]
fn FileTree(node: FileNode) -> impl IntoView {
    let has_children = node.children.is_some() && !node.children.as_ref().unwrap().is_empty();
    let children = node.children.clone().unwrap_or_default();

    view! {
        <div class="file-node" style="margin-left: 10px; border-left: 1px solid #eee; padding-left: 5px;">
            <div class="node-label" style=format!("font-weight: {}; color: {}; cursor: pointer; padding: 2px 0;", if node.is_dir { "bold" } else { "normal" }, if node.is_dir { "#000" } else { "#666" })>
                {if node.is_dir { "📁 " } else { "📄 " }} {node.name}
            </div>
            {if has_children {
                view! {
                    <div class="node-children">
                        <For
                            each=move || children.clone()
                            key=|child| format!("{}-{}", child.name, child.is_dir)
                            children=|child| {
                                view! { <FileTree node=child /> }
                            }
                        />
                    </div>
                }.into_any()
            } else {
                ().into_any()
            }}
        </div>
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    response: String,
}

#[component]
pub fn App() -> impl IntoView {
    let (input, set_input) = signal(String::new());
    let (messages, set_messages) = signal(Vec::<ChatMessage>::new());
    let (is_loading, set_is_loading) = signal(false);

    let send_message = move || {
        let current_input = input.get();
        if current_input.is_empty() || is_loading.get() {
            return;
        }

        // Add user message
        set_messages.update(|msgs| {
            msgs.push(ChatMessage {
                role: "User".to_string(),
                content: current_input.clone(),
            });
        });
        
        let prompt = current_input.clone();
        set_input.set(String::new());
        set_is_loading.set(true);

        spawn_local(async move {
            let payload = serde_json::json!({
                "model": "qwen3:8b",
                "prompt": prompt
            });

            match Request::post("http://127.0.0.1:3000/chat")
                .json(&payload)
                .unwrap()
                .send()
                .await
            {
                Ok(resp) => {
                    if let Ok(data) = resp.json::<ChatResponse>().await {
                        set_messages.update(|msgs| {
                            msgs.push(ChatMessage {
                                role: "TWAI".to_string(),
                                content: data.response,
                            });
                        });
                    } else {
                        set_messages.update(|msgs| {
                            msgs.push(ChatMessage {
                                role: "System".to_string(),
                                content: "Error parsing response".to_string(),
                            });
                        });
                    }
                }
                Err(e) => {
                    set_messages.update(|msgs| {
                        msgs.push(ChatMessage {
                            role: "System".to_string(),
                            content: format!("Network Error: {:?}", e),
                        });
                    });
                }
            }
            
            set_is_loading.set(false);
        });
    };

    view! {
        <div class="chat-container" style="display: flex; flex-direction: column; height: 100%;">
            <div class="messages" style="flex: 1; overflow-y: auto; border: 2px solid #000; padding: 15px; background: #fafafa; margin-bottom: 15px; font-family: 'Courier New', monospace;">
                <For
                    each=move || messages.get()
                    key=|msg| format!("{}-{}", msg.role, msg.content.len())
                    children=|msg| {
                        let role = msg.role.clone();
                        let content = msg.content.clone();
                        view! {
                            <div class=format!("message {}", role) style="margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #ddd;">
                                <strong style="text-transform: uppercase; color: #555;">{role.clone()}: </strong> 
                                <span style="white-space: pre-wrap;">{content}</span>
                            </div>
                        }
                    }
                />
            </div>
            <div class="input-area" style="display: flex; gap: 10px;">
                <input
                    type="text"
                    style="flex: 1; padding: 12px; border: 2px solid #000; outline: none; font-family: inherit;"
                    placeholder="Enter command or query..."
                    prop:value=input
                    on:input=move |ev| set_input.set(event_target_value(&ev))
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" {
                            send_message();
                        }
                    }
                />
                <button
                    style="padding: 10px 25px; background: #000; color: #fff; border: none; cursor: pointer; font-weight: bold; text-transform: uppercase;"
                    on:click=move |_| send_message()
                    prop:disabled=move || is_loading.get()
                >
                    {move || if is_loading.get() { "..." } else { "Execute" }}
                </button>
            </div>
        </div>
    }
}

fn main() {
    console_error_panic_hook::set_once();
    
    let doc = web_sys::window().unwrap().document().unwrap();
    
    if let Some(el) = doc.get_element_by_id("leptos-app") {
        leptos::mount::mount_to(el.unchecked_into(), App).forget();
    }

    if let Some(el) = doc.get_element_by_id("map-placeholder") {
        leptos::mount::mount_to(el.unchecked_into(), || {
            let (project_map, set_project_map) = signal(None::<FileNode>);
            let (path_input, set_path_input) = signal(String::new());
            let (status_msg, set_status_msg) = signal(String::from("Ready."));

            let fetch_map = move || {
                spawn_local(async move {
                    set_status_msg.set("Fetching map...".to_string());
                    match Request::get("http://127.0.0.1:3000/api/map").send().await {
                        Ok(resp) => {
                            if let Ok(map) = resp.json::<FileNode>().await {
                                set_project_map.set(Some(map));
                                set_status_msg.set("Map loaded.".to_string());
                            } else {
                                set_status_msg.set("Error: Failed to parse map data.".to_string());
                            }
                        }
                        Err(e) => {
                            set_status_msg.set(format!("Error: {:?}", e));
                        }
                    }
                });
            };

            let set_root = move |_| {
                let path = path_input.get();
                set_status_msg.set(format!("Setting root to: {}...", path));
                spawn_local(async move {
                    match Request::post("http://127.0.0.1:3000/api/set_root")
                        .json(&serde_json::json!({ "path": path }))
                        .unwrap()
                        .send()
                        .await 
                    {
                        Ok(_) => {
                            set_status_msg.set("Root set. Reloading map...".to_string());
                            fetch_map();
                        }
                        Err(e) => {
                            set_status_msg.set(format!("Error setting root: {:?}", e));
                        }
                    }
                });
            };

            Effect::new(move |_| fetch_map());

            view! {
                <div class="map-controls" style="margin-bottom: 15px;">
                    <input 
                        type="text" 
                        placeholder="Project path..." 
                        style="width: 100%; box-sizing: border-box; padding: 5px; border: 1px solid #000; margin-bottom: 5px;"
                        prop:value=path_input
                        on:input=move |ev| set_path_input.set(event_target_value(&ev))
                    />
                    <button 
                        style="width: 100%; padding: 5px; background: #eee; border: 1px solid #000; cursor: pointer; font-size: 0.8rem;"
                        on:click=set_root
                    >
                        "Load Folder"
                    </button>
                    <div style="color: #666; font-size: 0.7rem; margin-top: 5px; min-height: 1.2em;">
                        {move || status_msg.get()}
                    </div>
                </div>
                <div class="map-container" style="font-size: 0.8rem; font-family: monospace; max-height: 600px; overflow-y: auto;">
                    {move || match project_map.get() {
                        Some(map) => view! { <FileTree node=map /> }.into_any(),
                        None => view! { <div></div> }.into_any(),
                    }}
                </div>
            }
        }).forget();
    }
}