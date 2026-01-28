const fileInput = document.getElementById('file');
const uploadBtn = document.getElementById('upload-btn');
const fileStatus = document.getElementById('file-status');
const filenameSpan = document.getElementById('filename');
const statusLabel = document.getElementById('status-label');
const statusDot = document.getElementById('status-dot');
const resetBtn = document.getElementById('reset-btn');
const chatHistory = document.getElementById('chat-history');
const questionInput = document.getElementById('question');
const themeToggle = document.getElementById('theme-toggle');

let sessionId = null;

// Theme Logic with Persistence
const savedTheme = localStorage.getItem('theme');
// Default to dark if null, otherwise respect saved choice
if (!savedTheme || savedTheme === 'dark') {
  document.body.classList.add('dark-mode');
}

themeToggle.addEventListener('click', () => {
  document.body.classList.toggle('dark-mode');
  const isDark = document.body.classList.contains('dark-mode');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

// Upload
fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // UI: Uploading State
  fileStatus.classList.remove('hidden');
  filenameSpan.textContent = file.name;
  statusLabel.textContent = "Uploading...";
  statusDot.className = "status-dot loading";
  uploadBtn.disabled = true;

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    sessionId = data.session_id;

    // UI: Ready State
    statusLabel.textContent = "Indexed";
    statusDot.className = "status-dot"; // Stop animation

    questionInput.disabled = false;
    questionInput.placeholder = "Message RAG...";
    questionInput.focus();

    resetBtn.disabled = false; // Enable Reset
    uploadBtn.classList.add('hidden');

    // Clear history
    chatHistory.innerHTML = '';
    appendMessage('system', `Document "${file.name}" is ready for questions.`);

  } catch (e) {
    statusLabel.textContent = "Failed";
    statusDot.style.background = "red";
    statusDot.className = "status-dot";
    uploadBtn.disabled = false;
  }
});

// Reset Logic (Soft Reset)
async function reset() {
  if (!sessionId) return;

  // Notify backend
  navigator.sendBeacon(`/reset?session_id=${sessionId}`);
  sessionId = null;

  // Reset UI State
  chatHistory.innerHTML = '<div class="empty-state"><p>Session reset. Upload a new document.</p></div>';

  fileInput.value = ''; // Clear file input
  filenameSpan.textContent = '';
  fileStatus.classList.add('hidden');

  questionInput.disabled = true;
  questionInput.placeholder = "Ask a question...";

  resetBtn.disabled = true; // Disable instead of hide
  uploadBtn.classList.remove('hidden'); // Show upload button again
  uploadBtn.disabled = false;
  uploadBtn.textContent = '+ New Document';
}

resetBtn.addEventListener('click', reset);

// Chat
async function ask() {
  const q = questionInput.value.trim();
  if (!q) return;

  appendMessage('user', q);
  questionInput.value = '';

  const { content, metrics } = appendMessage('ai', '');
  content.classList.add('cursor');

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, question: q })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line) continue;
        try {
          const data = JSON.parse(line);
          if (data.token) {
            content.textContent += data.token;
            chatHistory.scrollTop = chatHistory.scrollHeight;
          } else if (data.done) {
            metrics.innerHTML = `Latency: ${data.latency_ms}ms  •  Context: ${data.retrieved_chunks} chunks`;
          }
        } catch (e) { }
      }
    }
  } catch (e) {
    content.textContent = 'Connection Error';
  } finally {
    content.classList.remove('cursor');
  }
}

function appendMessage(role, text) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const content = document.createElement('div');
  content.className = 'content';
  content.textContent = text;
  msg.appendChild(content);

  let metrics = null;
  if (role === 'ai') {
    metrics = document.createElement('div');
    metrics.className = 'metrics-box';
    msg.appendChild(metrics);
  }

  chatHistory.appendChild(msg);
  chatHistory.scrollTop = chatHistory.scrollHeight;

  return { content, metrics };
}

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') ask();
});
