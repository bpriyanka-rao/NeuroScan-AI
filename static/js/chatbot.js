// Chatbot UI Toggle
function toggleChatbot() {
  const panel = document.getElementById('chatbot-panel');
  const widget = document.getElementById('chatbot-widget');
  
  if (panel.classList.contains('hidden')) {
    panel.classList.remove('hidden');
    widget.classList.remove('chatbot-closed');
    document.getElementById('chatbot-input').focus();
  } else {
    panel.classList.add('hidden');
    widget.classList.add('chatbot-closed');
  }
}

// Handle Enter key in input
function handleChatKeypress(event) {
  if (event.key === 'Enter') {
    sendChatMessage();
  }
}

// Append message to chat
function appendMessage(text, isUser = false) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
  const messageText = document.createElement('span');
  messageText.textContent = text;
  msgDiv.appendChild(messageText);

  if (!isUser && 'speechSynthesis' in window) {
    const speakButton = document.createElement('button');
    speakButton.type = 'button';
    speakButton.className = 'chat-speak-btn';
    speakButton.title = 'Speak response';
    speakButton.setAttribute('aria-label', 'Speak response');
    speakButton.innerHTML = '<i class="fa fa-volume-high"></i>';
    speakButton.addEventListener('click', () => speakChatMessage(text));
    msgDiv.appendChild(speakButton);
  }
  
  const container = document.getElementById('chatbot-messages');
  // Remove typing indicator if exists
  const typing = document.getElementById('typing-indicator');
  if (typing) {
    container.removeChild(typing);
  }
  
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function speakChatMessage(text) {
  if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
    showToast('Speech playback is not supported in this browser.', 'error');
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  utterance.onerror = () => showToast('Could not play this response aloud.', 'error');
  window.speechSynthesis.speak(utterance);
}

// Show typing indicator
function showTypingIndicator() {
  const container = document.getElementById('chatbot-messages');
  const typingDiv = document.createElement('div');
  typingDiv.id = 'typing-indicator';
  typingDiv.className = 'message ai-message typing';
  typingDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
  container.appendChild(typingDiv);
  container.scrollTop = container.scrollHeight;
}

// Send message to backend API
async function sendChatMessage() {
  const inputEl = document.getElementById('chatbot-input');
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = '';
  appendMessage(text, true);
  showTypingIndicator();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    
    const data = await response.json();
    if (data.reply) {
      appendMessage(data.reply, false);
    } else {
      appendMessage('Error: ' + (data.error || 'Unknown error'), false);
    }
  } catch (error) {
    appendMessage('Sorry, I am having trouble connecting to the server.', false);
  }
}

// Web Speech API (Voice Input)
function startVoiceInput() {
  const micBtn = document.getElementById('chatbot-mic-btn');
  const inputEl = document.getElementById('chatbot-input');
  
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    if (typeof showToast === 'function') {
      showToast('Voice input is not supported in this browser.', 'error');
    } else {
      alert('Voice input is not supported in this browser.');
    }
    return;
  }
  
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  
  recognition.onstart = function() {
    micBtn.style.color = 'var(--red)';
    inputEl.placeholder = 'Listening...';
  };
  
  recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    inputEl.value = transcript;
    inputEl.placeholder = 'Type your question...';
    // Optional: auto-send after voice input
    // sendChatMessage(); 
  };
  
  recognition.onerror = function(event) {
    inputEl.placeholder = 'Type your question...';
    micBtn.style.color = '';
    console.error('Speech recognition error', event.error);
    const messages = {
      'not-allowed': 'Allow microphone access and use localhost or an HTTPS address.',
      'audio-capture': 'No microphone was detected.',
      'no-speech': 'No speech was detected. Try again.',
      'network': 'Voice recognition is unavailable right now.',
    };
    showToast(messages[event.error] || 'Voice input could not start.', 'error');
  };
  
  recognition.onend = function() {
    micBtn.style.color = '';
    inputEl.placeholder = 'Type your question...';
  };
  
  try {
    recognition.start();
  } catch (error) {
    inputEl.placeholder = 'Type your question...';
    micBtn.style.color = '';
    showToast('Voice input could not start. Check microphone permission.', 'error');
  }
}
