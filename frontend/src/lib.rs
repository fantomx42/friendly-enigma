use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use gloo_net::eventsource::futures::EventSource;
#[allow(unused_imports)]
use futures::StreamExt;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ChatMessage {
    role: String,
    content: String,
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
            // Future integration with backend
            set_is_loading.set(false);
        });
    };

    view! {
        <div class="chat-container">
            <div class="messages" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;">
                <For
                    each=move || messages.get()
                    key=|msg| format!("{}-{}", msg.role, msg.content)
                    children=|msg| {
                        let role = msg.role.clone();
                        let content = msg.content.clone();
                        view! {
                            <div class=format!("message {}", role)>
                                <strong>{role.clone()}: </strong> {content}
                            </div>
                        }
                    }
                />
            </div>
            <div class="input-area" style="display: flex; gap: 10px;">
                <input
                    type="text"
                    style="flex: 1; padding: 10px;"
                    prop:value=input
                    on:input=move |ev| set_input.set(event_target_value(&ev))
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" {
                            send_message();
                        }
                    }
                />
                <button
                    style="padding: 10px 20px; background: #000; color: #fff; border: none; cursor: pointer;"
                    on:click=move |_| send_message()
                    prop:disabled=move || is_loading.get()
                >
                    {move || if is_loading.get() { "..." } else { "Send" }}
                </button>
            </div>
        </div>
    }
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(App);
}

#[cfg(test)]
mod tests {
    #[test]
    fn frontend_compiles() {
        assert_eq!(1 + 1, 2);
    }
}