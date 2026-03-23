const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');
const processLog = document.getElementById('process-log');
const statusIndicator = document.getElementById('status-indicator');

let isProcessing = false;

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message || isProcessing) return;

    // UI Updates
    addMessage(message, 'user');
    userInput.value = '';
    isProcessing = true;
    statusIndicator.textContent = 'Processing...';
    processLog.innerHTML = ''; // Clear previous process logs

    // Create a placeholder for the assistant's response
    const assistantMessageDiv = createMessageElement('system');
    chatHistory.appendChild(assistantMessageDiv);
    const bubble = assistantMessageDiv.querySelector('.bubble');
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    scrollToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const data = JSON.parse(line);

                    if (data.type === 'step') {
                        addProcessStep(data.content);
                    } else if (data.type === 'data') {
                        addProcessData(data.title, data.data);
                    } else if (data.type === 'rag_context') {
                        addProcessRagContext(data.title, data.chunks);
                    } else if (data.type === 'result') {
                        // Replace typing indicator with actual text
                        if (assistantText === '') {
                            bubble.innerHTML = '';
                        }
                        assistantText = data.content;
                        // For a smoother effect, we could stream char by char, but for now just replacing is fine
                        // or we can append if the backend sends chunks of text
                        bubble.innerHTML = formatResponse(assistantText);
                        scrollToBottom();
                    } else if (data.type === 'error') {
                        addProcessStep(`Error: ${data.content}`, true);
                        if (assistantText === '') {
                            bubble.innerHTML = `<span style="color:red">Error: ${data.content}</span>`;
                        }
                    }

                } catch (jsonError) {
                    console.error('Error parsing JSON chunk', jsonError);
                }
            }
        }

    } catch (error) {
        console.error('Fetch error:', error);
        addProcessStep('Network Error occurred.', true);
        bubble.innerHTML = '<span style="color:red">Failed to send message.</span>';
    } finally {
        isProcessing = false;
        statusIndicator.textContent = 'Idle';
    }
});

function createMessageElement(sender) {
    const div = document.createElement('div');
    div.classList.add('message', sender);

    // Avatar
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    if (sender === 'system') {
        avatar.classList.add('system-avatar');
        avatar.innerText = 'AI';
        // HTML for sparkle icon or just text
        avatar.innerHTML = '<i class="fas fa-sparkles"></i>';
        // Fallback if no icon font:
        if (!avatar.querySelector('i')) avatar.innerText = 'AI';
    } else {
        avatar.classList.add('user-avatar');
        avatar.innerText = 'U';
    }

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');

    div.appendChild(avatar);
    div.appendChild(bubble);
    return div;
}

function addMessage(text, sender) {
    const div = createMessageElement(sender);
    div.querySelector('.bubble').innerText = text;
    chatHistory.appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    const container = document.querySelector('.chat-history-container');
    container.scrollTop = container.scrollHeight;
}

function addProcessStep(text, isError = false) {
    const step = document.createElement('div');
    step.classList.add('step-item');
    if (isError) step.style.borderLeftColor = 'red';
    step.innerText = text;
    processLog.appendChild(step);
    // processLog is now inside a container that scrolls
    if (processLog.parentElement) {
        processLog.parentElement.scrollTop = processLog.parentElement.scrollHeight;
    }
}

function addProcessData(title, dataObj) {
    const container = document.createElement('div');
    container.classList.add('step-data');

    const header = document.createElement('strong');
    header.style.display = 'block';
    header.style.marginBottom = '5px';
    header.innerText = title;
    container.appendChild(header);

    for (const [key, value] of Object.entries(dataObj)) {
        const row = document.createElement('div');
        row.classList.add('data-row');
        row.innerHTML = `<span class="data-label">${key}:</span> <span class="data-value">${value}</span>`;
        container.appendChild(row);
    }

    // Append to the last step item if possible, or just to the log
    const lastStep = processLog.lastElementChild;
    if (lastStep && lastStep.classList.contains('step-item')) {
        lastStep.appendChild(container);
    } else {
        processLog.appendChild(container); // Fallback
    }
    if (processLog.parentElement) {
        processLog.parentElement.scrollTop = processLog.parentElement.scrollHeight;
    }
}

function addProcessRagContext(title, chunks) {
    const container = document.createElement('div');
    container.classList.add('step-data', 'rag-container');

    const header = document.createElement('strong');
    header.style.display = 'block';
    header.style.marginBottom = '8px';
    header.innerText = title;
    container.appendChild(header);

    // store chunks for modal
    window.currentRagChunks = chunks;

    chunks.forEach((chunk, index) => {
        const sourceCard = document.createElement('div');
        sourceCard.classList.add('rag-source-card');
        
        let metaHtml = '';
        if (chunk.metadata && Object.keys(chunk.metadata).length > 0) {
            metaHtml = `
                <div class="rag-metadata-tags">
                    <span class="rag-tag jurisdiction-tag">${chunk.metadata.jurisdiction || 'Unknown'}</span>
                    <span class="rag-tag category-tag">${chunk.metadata.category || 'Law'}</span>
                </div>
                <div class="rag-law-title">${chunk.metadata.title || ''}</div>
                <div class="rag-law-source"><i>Source: ${chunk.metadata.source_file || 'Unknown'}</i></div>
            `;
        }
        
        sourceCard.innerHTML = `
            <div class="rag-source-header" onclick="this.parentElement.classList.toggle('expanded')">
               <span class="rag-source-title">Source ${index + 1} (Score: ${chunk.score.toFixed(2)})</span>
               <i class="fas fa-chevron-down toggle-icon"></i>
            </div>
            <div class="rag-source-content">
                ${metaHtml}
                <div class="rag-text-preview">${chunk.text}</div>
            </div>
        `;
        container.appendChild(sourceCard);
    });

    // Add "View in Modal" button
    const viewBtnDiv = document.createElement('div');
    viewBtnDiv.classList.add('rag-source-card-actions');
    viewBtnDiv.innerHTML = `<button class="rag-view-btn" onclick="openRagModal()"><i class="fas fa-expand-arrows-alt"></i> View Retrieved Laws</button>`;
    container.appendChild(viewBtnDiv);

    const lastStep = processLog.lastElementChild;
    if (lastStep && lastStep.classList.contains('step-item')) {
        lastStep.appendChild(container);
    } else {
        processLog.appendChild(container);
    }
    if (processLog.parentElement) {
        processLog.parentElement.scrollTop = processLog.parentElement.scrollHeight;
    }
}

function formatResponse(text) {
    // Simple markdown-to-html converter (very basic)
    // Replace \n with <br>
    // Bold **text** with <b>text</b>
    let formatted = text
        .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
        .replace(/\n/g, '<br>');
    return formatted;
}

// --- Modal Logic ---
const ragModal = document.getElementById('rag-modal');
const closeModalBtn = document.getElementById('close-modal-btn');
const modalRagContent = document.getElementById('modal-rag-content');

if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => {
        ragModal.classList.add('hidden');
    });
}

// Close when clicking outside of modal container
if (ragModal) {
    ragModal.addEventListener('click', (e) => {
        if (e.target === ragModal) {
            ragModal.classList.add('hidden');
        }
    });
}

function openRagModal() {
    if (!window.currentRagChunks || window.currentRagChunks.length === 0) return;
    
    modalRagContent.innerHTML = '';
    
    window.currentRagChunks.forEach((chunk, i) => {
        const div = document.createElement('div');
        div.classList.add('modal-rag-item');
        
        let metaHtml = '';
        if (chunk.metadata && Object.keys(chunk.metadata).length > 0) {
            metaHtml = `
                <div class="modal-metadata-header">
                    <div class="rag-metadata-tags">
                        <span class="rag-tag jurisdiction-tag">${chunk.metadata.jurisdiction || 'Unknown'}</span>
                        <span class="rag-tag category-tag">${chunk.metadata.category || 'Law'}</span>
                    </div>
                    <div class="rag-law-title">${chunk.metadata.title || ''}</div>
                    <div class="rag-law-source"><i>Source: ${chunk.metadata.source_file || 'Unknown'}</i></div>
                </div>
            `;
        }
        
        div.innerHTML = `
            <div class="modal-rag-score"><span>Source ${i + 1}</span> <span>Score: ${(chunk.score || 0).toFixed(3)}</span></div>
            ${metaHtml}
            <div class="modal-rag-text">${formatResponse(chunk.text)}</div>
        `;
        modalRagContent.appendChild(div);
    });
    
    ragModal.classList.remove('hidden');
}
