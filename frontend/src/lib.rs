use leptos::prelude::*;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <div class="chat-container">
            <p>"Welcome, Architect. TWAI is initializing..."</p>
            <div class="status-indicator">"System Status: ONLINE"</div>
        </div>
    }
}

#[cfg(test)]

mod tests {

    #[test]

    fn frontend_compiles() {

        assert_eq!(1 + 1, 2);

    }

}
