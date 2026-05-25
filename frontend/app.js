const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const loading = document.querySelector("#loading");
const sendButton = document.querySelector("#sendButton");
const newChat = document.querySelector("#newChat");

function getSessionId() {
  let sessionId = localStorage.getItem("sessionId");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("sessionId", sessionId);
  }
  return sessionId;
}

function addMessage(role, text, sources = []) {
  const bubble = document.createElement("article");
  bubble.className = `message ${role}`;
  bubble.textContent = text;

  if (sources.length > 0) {
    const sourceLine = document.createElement("div");
    sourceLine.className = "sources";
    sourceLine.textContent = `Sources: ${sources
      .map((source) => `${source.title} (${source.score})`)
      .join(", ")}`;
    bubble.appendChild(sourceLine);
  }

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
}

async function sendMessage(message) {
  loading.hidden = false;
  sendButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: getSessionId(),
        message,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The chat request failed.");
    }

    addMessage("assistant", data.reply, data.sources || []);
  } catch (error) {
    addMessage("error", error.message);
  } finally {
    loading.hidden = true;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMessage("user", message);
  sendMessage(message);
});

newChat.addEventListener("click", () => {
  localStorage.removeItem("sessionId");
  messages.innerHTML = "";
  getSessionId();
  input.focus();
});

getSessionId();
addMessage(
  "assistant",
  "Ask me a question about the knowledge base. I will retrieve relevant chunks before answering."
);
