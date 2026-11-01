/* Spring AI RAG — minimal UI.
 *
 * Security note: every piece of server- or model-supplied text is inserted
 * with textContent, never innerHTML. Document contents and LLM output are
 * untrusted input as far as the DOM is concerned.
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const el = {
    dropzone: $('dropzone'),
    fileInput: $('file-input'),
    staged: $('staged'),
    stagedList: $('staged-list'),
    uploadBtn: $('upload-btn'),
    clearStaged: $('clear-staged'),
    uploadStatus: $('upload-status'),
    refreshBtn: $('refresh-btn'),
    docList: $('doc-list'),
    docEmpty: $('doc-empty'),
    statDocs: $('stat-docs'),
    statChunks: $('stat-chunks'),
    messages: $('messages'),
    chatForm: $('chat-form'),
    question: $('question'),
    sendBtn: $('send-btn'),
    toggleSettings: $('toggle-settings'),
    settings: $('settings'),
    topk: $('topk'),
    threshold: $('threshold'),
    examples: $('examples'),
    statProvider: $('stat-provider'),
    scopeSelect: $('scope-select'),
  };

  let stagedFiles = [];

  // ── helpers ──────────────────────────────────────────────

  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function setStatus(node, text, kind) {
    node.textContent = text;
    node.className = 'status' + (kind ? ` ${kind}` : '');
    node.hidden = false;
  }

  /** Turns a failed fetch Response into a useful Error message. */
  async function describeError(res) {
    let detail = '';
    try {
      const body = await res.json();
      // Our ChatExceptionHandler sends {error, upstreamStatus, hint}
      detail = body.hint || body.message || body.error || '';
    } catch {
      detail = await res.text().catch(() => '');
    }
    return new Error(detail ? `${res.status} — ${detail}` : `HTTP ${res.status}`);
  }

  // ── provider badge ───────────────────────────────────────

  async function refreshProvider() {
    try {
      const res = await fetch('/api/info');
      if (!res.ok) throw await describeError(res);
      const info = await res.json();
      el.statProvider.textContent = `${info.chatProvider} · ${info.chatModel}`;
      el.statProvider.classList.toggle('free', info.requiresApiKey === false);
      el.statProvider.title = `${info.chatImplementation} · embeddings: `
        + `${info.embeddingModel} (${info.embeddingDimensions}d) · `
        + (info.requiresApiKey ? 'needs an API key' : 'local, no API key');
    } catch {
      el.statProvider.textContent = 'provider unknown';
    }
  }

  // ── document library ─────────────────────────────────────

  async function refreshLibrary() {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) throw await describeError(res);
      const data = await res.json();

      el.statDocs.textContent = plural(data.documentCount, 'document');
      el.statChunks.textContent = plural(data.totalChunks, 'chunk');

      el.docList.replaceChildren();
      el.docEmpty.hidden = data.documents.length > 0;

      // Keep the scope selector in step with the library, preserving the
      // current choice if that document still exists.
      const previous = el.scopeSelect.value;
      el.scopeSelect.replaceChildren();
      const allOpt = document.createElement('option');
      allOpt.value = '';
      allOpt.textContent = 'all documents';
      el.scopeSelect.appendChild(allOpt);
      for (const doc of data.documents) {
        const opt = document.createElement('option');
        opt.value = doc.source;
        opt.textContent = doc.source;
        el.scopeSelect.appendChild(opt);
      }
      if (data.documents.some((d) => d.source === previous)) {
        el.scopeSelect.value = previous;
      }

      for (const doc of data.documents) {
        const li = document.createElement('li');

        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = doc.source;          // untrusted filename
        name.title = doc.source;

        const chunks = document.createElement('span');
        chunks.className = 'chunks';
        chunks.textContent = plural(doc.chunks, 'chunk');

        const del = document.createElement('button');
        del.className = 'del-btn';
        del.type = 'button';
        del.textContent = '×';
        del.title = `Remove ${doc.source}`;
        del.addEventListener('click', () => removeDocument(doc.source));

        li.append(name, chunks, del);
        el.docList.appendChild(li);
      }
    } catch (err) {
      setStatus(el.uploadStatus, `Could not load library: ${err.message}`, 'err');
    }
  }

  async function removeDocument(source) {
    if (!confirm(`Remove "${source}" and all of its chunks?`)) return;
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(source)}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw await describeError(res);
      const data = await res.json();
      setStatus(el.uploadStatus,
        `Removed "${source}" (${plural(data.removedChunks, 'chunk')}).`, 'ok');
      refreshLibrary();
    } catch (err) {
      setStatus(el.uploadStatus, `Delete failed: ${err.message}`, 'err');
    }
  }

  // ── staging & upload ─────────────────────────────────────

  function stageFiles(fileList) {
    for (const f of fileList) {
      if (!stagedFiles.some((s) => s.name === f.name && s.size === f.size)) {
        stagedFiles.push(f);
      }
    }
    renderStaged();
  }

  function renderStaged() {
    el.staged.hidden = stagedFiles.length === 0;
    el.stagedList.replaceChildren();

    stagedFiles.forEach((file, index) => {
      const li = document.createElement('li');

      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = file.name;
      name.title = file.name;

      const size = document.createElement('span');
      size.className = 'size';
      size.textContent = formatSize(file.size);

      const remove = document.createElement('button');
      remove.className = 'del-btn';
      remove.type = 'button';
      remove.textContent = '×';
      remove.addEventListener('click', () => {
        stagedFiles.splice(index, 1);
        renderStaged();
      });

      li.append(name, size, remove);
      el.stagedList.appendChild(li);
    });

    el.uploadBtn.textContent = stagedFiles.length > 1
      ? `Upload ${stagedFiles.length} files`
      : 'Upload';
  }

  async function uploadStaged() {
    if (stagedFiles.length === 0) return;

    const form = new FormData();
    // Field name must be "files" — matches @RequestParam("files") MultipartFile[]
    stagedFiles.forEach((f) => form.append('files', f, f.name));

    el.uploadBtn.disabled = true;
    setStatus(el.uploadStatus, 'Embedding and storing…');
    el.uploadStatus.prepend(Object.assign(document.createElement('span'),
      { className: 'spinner' }));

    try {
      const res = await fetch('/api/documents', { method: 'POST', body: form });
      if (!res.ok) throw await describeError(res);
      const data = await res.json();

      const failures = data.results.filter((r) => r.status === 'error');
      const summary = `Ingested ${plural(data.uploaded, 'file')} into `
        + `${plural(data.totalChunks, 'chunk')}.`;

      if (failures.length > 0) {
        const detail = failures.map((f) => `${f.source}: ${f.error}`).join(' | ');
        setStatus(el.uploadStatus, `${summary} ${failures.length} failed — ${detail}`, 'err');
      } else {
        setStatus(el.uploadStatus, summary, 'ok');
      }

      stagedFiles = [];
      renderStaged();
      refreshLibrary();
    } catch (err) {
      setStatus(el.uploadStatus, `Upload failed: ${err.message}`, 'err');
    } finally {
      el.uploadBtn.disabled = false;
    }
  }

  // ── chat ─────────────────────────────────────────────────

  function clearWelcome() {
    const welcome = el.messages.querySelector('.welcome');
    if (welcome) welcome.remove();
  }

  function addMessage(role, text) {
    clearWelcome();
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;

    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = role === 'user' ? 'You' : 'Assistant';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;                  // untrusted — never innerHTML

    wrap.append(who, bubble);
    el.messages.appendChild(wrap);
    el.messages.scrollTop = el.messages.scrollHeight;
    return { wrap, bubble };
  }

  function addTyping() {
    const { wrap, bubble } = addMessage('bot', '');
    bubble.className = 'bubble typing';
    bubble.append(
      document.createElement('span'),
      document.createElement('span'),
      document.createElement('span'));
    return wrap;
  }

  function renderAnswer(wrap, data) {
    wrap.replaceChildren();

    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = 'Assistant';

    const badge = document.createElement('span');
    const fullDoc = data.strategy === 'FULL_DOCUMENT';
    badge.className = 'badge ' + (data.grounded ? 'grounded' : 'ungrounded');
    badge.textContent = fullDoc
      ? `full document · ${data.chunksUsed}/${data.chunksTotal} chunks`
      : (data.grounded ? `top matches · ${plural(data.citations.length, 'source')}`
                       : 'no sources retrieved');
    who.appendChild(badge);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = data.answer || '(empty response)';

    wrap.append(who, bubble);

    if (!data.grounded) {
      const note = document.createElement('div');
      note.className = 'ungrounded-note';
      note.textContent =
        `Nothing cleared the similarity threshold (${data.threshold}), so no `
        + `document text was sent to the model. Treat this answer with suspicion: `
        + `either your documents do not cover the question, or the threshold is too high.`;
      bubble.after(note);
    }

    if (data.citations && data.citations.length > 0) {
      const details = document.createElement('details');
      details.className = 'citations';

      const summary = document.createElement('summary');
      summary.textContent = `Sources (${data.citations.length}) — what grounded this answer`;
      details.appendChild(summary);

      for (const c of data.citations) {
        const cite = document.createElement('div');
        cite.className = 'cite';

        const head = document.createElement('div');
        head.className = 'cite-head';

        const score = document.createElement('span');
        score.className = 'score';
        score.textContent = c.score == null ? 'n/a' : Number(c.score).toFixed(4);

        const src = document.createElement('span');
        src.textContent = c.source;
        const chunk = document.createElement('span');
        chunk.textContent = `chunk #${c.chunkIndex}`;

        head.append(score, src, chunk);

        const text = document.createElement('div');
        text.className = 'cite-text';
        text.textContent = c.excerpt;

        cite.append(head, text);
        details.appendChild(cite);
      }
      wrap.appendChild(details);
    }

    // Why this retrieval path was chosen — otherwise the routing is invisible.
    if (data.strategyReason) {
      const why = document.createElement('p');
      why.className = 'strategy-note' + (data.truncated ? ' warn' : '');
      why.textContent = data.truncated
        ? `TRUNCATED — only ${data.chunksUsed} of ${data.chunksTotal} chunks were sent`
        : data.strategyReason;
      wrap.appendChild(why);
    }

    if (data.usage && data.usage.totalTokens) {
      const usage = document.createElement('div');
      usage.className = 'usage';
      usage.textContent =
        `${data.usage.promptTokens} in · ${data.usage.completionTokens} out · `
        + `${data.usage.totalTokens} total tokens · topK=${data.topK} · threshold=${data.threshold}`;
      wrap.appendChild(usage);
    }

    el.messages.scrollTop = el.messages.scrollHeight;
  }

  async function ask(question) {
    addMessage('user', question);
    const pending = addTyping();
    el.sendBtn.disabled = true;

    const body = {
      question,
      source: el.scopeSelect.value || undefined,
      topK: Number(el.topk.value) || undefined,
      threshold: el.threshold.value === '' ? undefined : Number(el.threshold.value),
    };

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw await describeError(res);
      renderAnswer(pending, await res.json());
    } catch (err) {
      pending.className = 'msg bot error';
      pending.replaceChildren();
      const who = document.createElement('div');
      who.className = 'who';
      who.textContent = 'Error';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = err.message;
      pending.append(who, bubble);
    } finally {
      el.sendBtn.disabled = false;
      el.question.focus();
    }
  }

  // ── wiring ───────────────────────────────────────────────

  el.dropzone.addEventListener('click', () => el.fileInput.click());
  el.dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
  });
  el.fileInput.addEventListener('change', (e) => {
    stageFiles(e.target.files);
    e.target.value = '';           // allow re-selecting the same file
  });

  ['dragenter', 'dragover'].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.add('dragover');
    }));
  ['dragleave', 'drop'].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove('dragover');
    }));
  el.dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer?.files?.length) stageFiles(e.dataTransfer.files);
  });

  el.uploadBtn.addEventListener('click', uploadStaged);
  el.clearStaged.addEventListener('click', () => { stagedFiles = []; renderStaged(); });
  el.refreshBtn.addEventListener('click', refreshLibrary);


  el.toggleSettings.addEventListener('click', () => {
    el.settings.hidden = !el.settings.hidden;
  });

  el.examples.addEventListener('click', (e) => {
    if (e.target.classList.contains('chip')) {
      el.question.value = e.target.textContent;
      el.chatForm.requestSubmit();
    }
  });

  el.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = el.question.value.trim();
    if (!q) return;
    el.question.value = '';
    ask(q);
  });

  refreshProvider();
  refreshLibrary();
})();
