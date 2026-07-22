const STORAGE_KEY = "mindpal_current_chat_v1";

const messagesEl = document.getElementById("messages");
const chatScrollEl = document.getElementById("chatScroll");
const welcomeCardEl = document.getElementById("welcomeCard");
const chatFormEl = document.getElementById("chatForm");
const messageInputEl = document.getElementById("messageInput");
const sendBtnEl = document.getElementById("sendBtn");
const newChatBtnEl = document.getElementById("newChatBtn");

let messages = [];
let isSending = false;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function autoResizeTextarea() {
  messageInputEl.style.height = "auto";
  messageInputEl.style.height = Math.min(messageInputEl.scrollHeight, 180) + "px";
}

function saveMessages() {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
}

function loadMessages() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function scrollToBottom(force = false) {
  requestAnimationFrame(() => {
    if (force) {
      chatScrollEl.scrollTop = chatScrollEl.scrollHeight;
      return;
    }
    const nearBottom =
      chatScrollEl.scrollHeight - chatScrollEl.scrollTop - chatScrollEl.clientHeight < 200;
    if (nearBottom) {
      chatScrollEl.scrollTop = chatScrollEl.scrollHeight;
    }
  });
}

function hideWelcomeIfNeeded() {
  if (!welcomeCardEl) return;
  welcomeCardEl.style.display = messages.length > 0 ? "none" : "block";
}

function renderMessage(message) {
  const role = message.role === "user" ? "user" : "assistant";
  const name = role === "user" ? "You" : "M";
  const meta = role === "user" ? "You" : "MindPal";

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "message-bubble-wrap";

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = name;

  const block = document.createElement("div");
  block.className = "bubble-block";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = formatMessageText(message.content || "");

  const metaEl = document.createElement("div");
  metaEl.className = "bubble-meta";
  metaEl.textContent = meta;

  block.appendChild(bubble);
  block.appendChild(metaEl);

  bubbleWrap.appendChild(avatar);
  bubbleWrap.appendChild(block);
  row.appendChild(bubbleWrap);

  messagesEl.appendChild(row);
}

function formatMessageText(text) {
  let safe = escapeHtml(text);

  safe = safe.replace(
    /^•\s(.+)$/gm,
    "<div>• $1</div>"
  );

  safe = safe.replace(/\n/g, "<br>");
  return safe;
}

function renderAllMessages() {
  const existing = messagesEl.querySelectorAll(".message-row, .typing-row");
  existing.forEach((node) => node.remove());

  hideWelcomeIfNeeded();

  messages.forEach((msg) => {
    renderMessage(msg);
  });

  scrollToBottom(true);
}

function setLoadingState(loading) {
  isSending = loading;
  sendBtnEl.disabled = loading;
  messageInputEl.disabled = loading;
}

function removeTypingIndicator() {
  const existing = document.getElementById("typingIndicator");
  if (existing) existing.remove();
}

function renderTypingIndicator() {
  removeTypingIndicator();

  const row = document.createElement("div");
  row.className = "message-row assistant typing-row";
  row.id = "typingIndicator";

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "message-bubble-wrap";

  const avatar = document.createElement("div");
  avatar.className = "avatar assistant";
  avatar.textContent = "M";

  const block = document.createElement("div");
  block.className = "bubble-block";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = `
    <span>MindPal is replying</span>
    <span class="typing-dots"><span></span><span></span><span></span></span>
  `;

  bubble.appendChild(typing);
  block.appendChild(bubble);
  bubbleWrap.appendChild(avatar);
  bubbleWrap.appendChild(block);
  row.appendChild(bubbleWrap);

  messagesEl.appendChild(row);
  scrollToBottom(true);
}

async function sendChatRequest(userMessage) {
  const payload = {
    message: userMessage,
    history: messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    })),
  };

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = "Something went wrong. Please try again.";
    try {
      const errorData = await response.json();
      detail = errorData.detail || detail;
    } catch {
      // ignore parse failure
    }
    throw new Error(detail);
  }

  return response.json();
}

async function handleSubmit(event) {
  event.preventDefault();

  if (isSending) return;

  const text = messageInputEl.value.trim();
  if (!text) return;

  const userMessage = {
    role: "user",
    content: text,
  };

  messages.push(userMessage);
  saveMessages();
  hideWelcomeIfNeeded();
  renderMessage(userMessage);
  scrollToBottom(true);

  messageInputEl.value = "";
  autoResizeTextarea();

  setLoadingState(true);
  renderTypingIndicator();

  try {
    const data = await sendChatRequest(text);
    removeTypingIndicator();

    const assistantText =
      data.answer ||
      data.response ||
      "I’m here with you 🌿";

    const assistantMessage = {
      role: "assistant",
      content: assistantText,
    };

    messages.push(assistantMessage);
    saveMessages();
    renderMessage(assistantMessage);
    scrollToBottom(true);
  } catch (error) {
    removeTypingIndicator();

    const assistantMessage = {
      role: "assistant",
      content:
        error.message ||
        "I’m sorry — I couldn’t respond properly just now. Please try again in a moment.",
    };

    messages.push(assistantMessage);
    saveMessages();
    renderMessage(assistantMessage);
    scrollToBottom(true);
  } finally {
    setLoadingState(false);
    messageInputEl.focus();
  }
}

async function startNewChat() {
  messages = [];
  saveMessages();
  renderAllMessages();
  messageInputEl.value = "";
  autoResizeTextarea();
  messageInputEl.focus();

  try {
    await fetch("/api/session/new", {
      method: "POST",
    });
  } catch {
    // backend endpoint optional
  }
}

function setupKeyboardSubmit() {
  messageInputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatFormEl.requestSubmit();
    }
  });
}

function initialize() {
  messages = loadMessages();
  renderAllMessages();
  autoResizeTextarea();
  setupKeyboardSubmit();

  chatFormEl.addEventListener("submit", handleSubmit);
  newChatBtnEl.addEventListener("click", startNewChat);
  messageInputEl.addEventListener("input", autoResizeTextarea);

  messageInputEl.focus();
}

initialize();