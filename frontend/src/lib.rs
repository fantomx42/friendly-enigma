use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use gloo_net::eventsource::futures::EventSource;
#[allow(unused_imports)]
use futures::StreamExt;
use gloo_net::http::Request;

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
        <div class="file-node" style="margin-left: 15px;">
            <div class="node-label" style=format!("font-weight: {}; color: {};", if node.is_dir { "bold" } else { "normal" }, if node.is_dir { "#000" } else { "#666" })>
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

        set_messages.update(|msgs| {
            msgs.push(ChatMessage {
                role: "user".to_string(),
                content: current_input.clone(),
            });
        });
        set_input.set(String::new());
        set_is_loading.set(true);

        spawn_local(async move {
            set_is_loading.set(false);
        });
    };

    view! {
        <div class="twai-container" style="display: contents;">
            <div class="chat-container" style="display: flex; flex-direction: column; height: 100%;">
                <div class="messages" style="flex: 1; overflow-y: auto; border: 2px solid #000; padding: 15px; background: #fafafa; margin-bottom: 15px; font-family: 'Courier New', monospace;">
                    <For
                        each=move || messages.get()
                        key=|msg| format!("{}-{}", msg.role, msg.content)
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
        </div>
    }
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn main() {
    use wasm_bindgen::JsCast;
    console_error_panic_hook::set_once();
    
    let doc = web_sys::window().unwrap().document().unwrap();
    
    // Mount Chat to the main area
    if let Some(el) = doc.get_element_by_id("leptos-app") {
        leptos::mount::mount_to(el.unchecked_into(), App).forget();
    }

    // Mount Map to the sidebar
    if let Some(el) = doc.get_element_by_id("map-placeholder") {
        leptos::mount::mount_to(el.unchecked_into(), || {
            let (project_map, set_project_map) = signal(None::<FileNode>);
            Effect::new(move |_| {
                spawn_local(async move {
                    if let Ok(resp) = Request::get("http://localhost:3000/api/map").send().await {
                        if let Ok(map) = resp.json::<FileNode>().await {
                            set_project_map.set(Some(map));
                        }
                    }
                });
            });

            view! {
                <div class="map-container" style="font-size: 0.9rem; font-family: monospace;">
                    {move || match project_map.get() {
                        Some(map) => view! { <FileTree node=map /> }.into_any(),
                        None => view! { <div>"Loading map..."</div> }.into_any(),
                    }}
                </div>
            }
        }).forget();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frontend_compiles() {
        assert_eq!(1 + 1, 2);
    }
}
