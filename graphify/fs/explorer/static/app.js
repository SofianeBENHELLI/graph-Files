// graphify fs explorer: browse, search and chat with the knowledge graph of your files.
import { h, render } from "preact";
import { useState, useEffect, useRef, useCallback, useMemo } from "preact/hooks";
import htm from "htm";

const html = htm.bind(h);

// ---------------------------------------------------------------- helpers
async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...opts,
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });
const qs = (o) => Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== "")
  .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");

const ICONS = { pdf: "📕", slides: "📊", document: "📄", spreadsheet: "📗", image: "🖼️", video: "🎬",
  audio: "🎧", text: "📝", web: "🌐", email: "✉️", archive: "🗜️", code: "💻", app: "🧩", other: "📦", folder: "📁" };
const KIND_LABEL = { pdf: "PDF", slides: "Slides", document: "Document", spreadsheet: "Spreadsheet", image: "Image",
  video: "Video", audio: "Audio", text: "Text", web: "Web", email: "Email", archive: "Archive", code: "Code",
  app: "App", other: "Other", folder: "Folder" };
const TAG_COLORS = { gray: "#9aa0a6", green: "#34a853", purple: "#a142f4", blue: "#4285f4", yellow: "#f2b600",
  red: "#ea4335", orange: "#fa7b17" };
const GROUP_COLORS = { pdf: "#e05252", slides: "#f08c2e", document: "#3c7be8", spreadsheet: "#2fa66a",
  image: "#a35be0", video: "#d64fa1", audio: "#c0478a", text: "#5d6b82", web: "#2aa3b8", other: "#8a93a3",
  code: "#5d6b82", archive: "#8a93a3", email: "#2aa3b8", tag: "#f2b600", concept: "#6d8dff", folder: "#9aa0a6",
  topic: "#2f5bea" };
const THUMB_KINDS = new Set(["pdf", "slides", "document", "spreadsheet", "image", "video", "text", "web", "code"]);

function fmtSize(n) {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"]; let i = -1;
  do { n /= 1024; i++; } while (n >= 1024 && i < u.length - 1);
  return `${n.toFixed(n < 10 ? 1 : 0)} ${u[i]}`;
}
function fmtDate(s) {
  if (!s) return "";
  const d = new Date(s); if (isNaN(d)) return s;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
const parentPath = (p) => (p && p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- pieces
function Thumb({ file, big }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [file.id]);
  const show = !failed && !file.offline && THUMB_KINDS.has(file.kind);
  return html`<div class="thumb" style=${big ? "height:auto;min-height:150px" : ""}>
    ${show ? html`<img loading="lazy" src=${`/api/thumb?${qs({ id: file.id, s: big ? 640 : 320 })}`} onError=${() => setFailed(true)} />`
           : html`<span class="ico">${ICONS[file.kind] || "📦"}</span>`}
    ${!big && html`<span class="badge">${file.offline ? "iCloud" : (KIND_LABEL[file.kind] || file.kind)}</span>`}
  </div>`;
}

function TagDot({ color }) {
  return html`<span class="dot" style=${color ? `background:${TAG_COLORS[color] || color}` : ""}></span>`;
}

function FileCard({ f, selected, onSelect, onOpen }) {
  return html`<div class=${"card" + (selected ? " sel" : "")} onClick=${() => onSelect(f.id)} onDblClick=${() => onOpen(f.id)} title=${f.path}>
    <${Thumb} file=${f} />
    <div class="meta">
      <div class="fname">${f.name}</div>
      <div class="fsub">${parentPath(f.path) || "/"} · ${fmtDate(f.modified)}</div>
      ${f.snippet ? html`<div class="snip">${renderSnippet(f.snippet)}</div>` : null}
      ${f.tags && f.tags.length ? html`<div class="chips">${f.tags.slice(0, 3).map((t) => html`<span class="chip mini">${t}</span>`)}</div>` : null}
    </div>
  </div>`;
}

function renderSnippet(text) {
  return text.split(/(«[^»]*»)/).map((part) => part.startsWith("«") ? html`<mark>${part.slice(1, -1)}</mark>` : part);
}

function FileTable({ files, selected, onSelect, onOpen }) {
  return html`<table class="list"><thead><tr><th>Name</th><th>Folder</th><th>Tags</th><th>Modified</th></tr></thead>
    <tbody>${files.map((f) => html`<tr class=${f.id === selected ? "sel" : ""} onClick=${() => onSelect(f.id)} onDblClick=${() => onOpen(f.id)}>
      <td><span class="ico">${ICONS[f.kind] || "📦"}</span>${f.name}${f.offline ? html` <span class="pill">iCloud</span>` : null}</td>
      <td style="color:var(--muted)">${parentPath(f.path) || "/"}</td>
      <td><div class="chips">${(f.tags || []).slice(0, 4).map((t) => html`<span class="chip mini">${t}</span>`)}</div></td>
      <td style="color:var(--muted);white-space:nowrap">${fmtDate(f.modified)}</td></tr>`)}</tbody></table>`;
}

function FolderTree({ path, depth, current, onPick, cache, load }) {
  const node = cache[path];
  useEffect(() => { if (!node) load(path); }, [path]);
  const [open, setOpen] = useState({});
  if (!node) return null;
  return html`<div>${node.folders.map((sub) => html`<div>
      <div class=${"item" + (current === sub.path ? " active" : "")} style=${`padding-left:${8 + depth * 14}px`}>
        <span class="caret" onClick=${(e) => { e.stopPropagation(); setOpen({ ...open, [sub.path]: !open[sub.path] }); }}>${open[sub.path] ? "▾" : "▸"}</span>
        <span class="name" onClick=${() => onPick(sub.path)}>📁 ${sub.name}</span><span class="count">${sub.files}</span>
      </div>
      ${open[sub.path] ? html`<${FolderTree} path=${sub.path} depth=${depth + 1} current=${current} onPick=${onPick} cache=${cache} load=${load} />` : null}
    </div>`)}</div>`;
}

function Section({ title, items, render: renderItem, limit = 12, filterable, extra }) {
  const [q, setQ] = useState("");
  const [all, setAll] = useState(false);
  const shown = useMemo(() => {
    const f = q ? items.filter((it) => (it.name || it.label || "").toLowerCase().includes(q.toLowerCase())) : items;
    return all || q ? f.slice(0, 400) : f.slice(0, limit);
  }, [items, q, all]);
  if (!items.length && !extra) return null;
  return html`<div class="sec"><h4>${title}${extra || null}</h4>
    ${filterable && items.length > limit ? html`<input class="filter" placeholder=${`Filter ${title.toLowerCase()}…`} value=${q} onInput=${(e) => setQ(e.target.value)} />` : null}
    ${shown.map(renderItem)}
    ${!q && items.length > limit ? html`<div class="more" onClick=${() => setAll(!all)}>${all ? "Show less" : `Show all ${items.length}`}</div>` : null}
  </div>`;
}

// ---------------------------------------------------------------- graph
function GraphView({ data, onNode, onNodeDbl, focusId }) {
  const ref = useRef(null);
  const net = useRef(null);
  useEffect(() => {
    if (!ref.current || !data || !window.vis) return;
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const nodes = data.nodes.map((n) => {
      const kind = n.fs_kind || "concept";
      const color = kind === "tag" && n.color ? (TAG_COLORS[n.color] || n.color) : (GROUP_COLORS[n.group] || GROUP_COLORS[kind] || "#8a93a3");
      const shape = kind === "file" ? "dot" : kind === "tag" ? "diamond" : n.group === "topic" ? "hexagon" : kind === "folder" ? "square" : "dot";
      return { id: n.id, label: n.label, title: n.title, shape, value: n.value || 1,
        color: { background: color, border: n.id === data.center ? (dark ? "#fff" : "#111") : color, highlight: { background: color, border: "#111" } },
        borderWidth: n.id === data.center ? 3 : 1, font: { color: dark ? "#e6e9ef" : "#1c2330", size: kind === "concept" ? 12 : 13 } };
    });
    const edges = data.edges.map((e, i) => ({ id: i, from: e.from, to: e.to, dashes: !!e.dashes, value: e.value || 1,
      title: e.relation, color: { color: dark ? "#3a4252" : "#cfd5df", highlight: "#2f5bea" },
      arrows: e.relation === "contains" || e.relation === "mentions" ? { to: { enabled: true, scaleFactor: 0.4 } } : undefined }));
    const ds = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
    const options = {
      physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -60, springLength: 90 },
        stabilization: { iterations: 250 } },
      nodes: { scaling: { min: 8, max: 34, label: { enabled: false } } },
      edges: { smooth: false, scaling: { min: 1, max: 5 } },
      interaction: { hover: true, tooltipDelay: 150 },
    };
    if (net.current) net.current.destroy();
    net.current = new vis.Network(ref.current, ds, options);
    net.current.on("click", (p) => { if (p.nodes.length) onNode(p.nodes[0], data.nodes.find((n) => n.id === p.nodes[0])); });
    net.current.on("doubleClick", (p) => { if (p.nodes.length) onNodeDbl(p.nodes[0], data.nodes.find((n) => n.id === p.nodes[0])); });
    net.current.once("stabilizationIterationsDone", () => net.current && net.current.setOptions({ physics: { enabled: false } }));
    return () => { if (net.current) { net.current.destroy(); net.current = null; } };
  }, [data]);
  useEffect(() => { if (net.current && focusId) { try { net.current.selectNodes([focusId]); } catch (_) { /* not in view */ } } }, [focusId, data]);
  if (!window.vis) return html`<div class="empty">Graph library failed to load.</div>`;
  return html`<div class="graphwrap"><div class="graph" ref=${ref}></div>
    <div class="legend">
      <span><span class="dot" style="background:${GROUP_COLORS.document}"></span>file</span>
      <span><span class="dot" style="background:${GROUP_COLORS.concept}"></span>concept</span>
      <span>◆ tag</span><span>⬢ topic</span><span>■ folder</span><span>┄ suggested tag</span>
    </div>
    <div class="graph-hint">click: open · double-click: explore around it</div></div>`;
}

// ---------------------------------------------------------------- inspector
function Inspector({ fileId, onSelect, onFocus, onView, onClose, toast, onTagsChanged, config }) {
  const [info, setInfo] = useState(null);
  const [preview, setPreview] = useState(false);
  const [text, setText] = useState(null);
  const [newTag, setNewTag] = useState("");
  const [err, setErr] = useState(null);
  const load = useCallback(() => {
    setErr(null);
    api(`/api/file?${qs({ id: fileId })}`).then(setInfo).catch((e) => setErr(e.message));
  }, [fileId]);
  useEffect(() => { setInfo(null); setPreview(false); setText(null); load(); }, [fileId]);
  useEffect(() => {
    if (preview && info && ["text", "code", "web"].includes(info.kind)) {
      fetch(`/api/raw?${qs({ id: info.id })}`, { credentials: "same-origin" }).then((r) => r.text()).then(setText).catch(() => setText("(cannot preview)"));
    }
  }, [preview, info]);
  if (err) return html`<div class="insp"><div class="note">${err}</div></div>`;
  if (!info) return html`<div class="insp"><div class="note">Loading…</div></div>`;
  const act = (action) => post("/api/open", { id: info.id, action }).catch((e) => toast(e.message));
  const tagAction = async (name, action) => {
    try {
      const r = await post("/api/tag", { id: info.id, name, action });
      setInfo({ ...info, tags: r.tags });
      toast(`${action === "reject" ? "Rejected" : action === "remove" ? "Removed" : "Tagged"} "${name}"${r.finder ? " · " + r.finder : ""}`);
      onTagsChanged && onTagsChanged();
    } catch (e) { toast(e.message); }
  };
  const canPreview = !info.offline && ["pdf", "image", "video", "audio", "text", "code", "web"].includes(info.kind);
  const raw = `/api/raw?${qs({ id: info.id })}`;
  return html`<div class="insp">
    <div class="insp-head"><h3>${ICONS[info.kind] || "📦"} ${info.name}</h3>
      <button class="btn ghost small" title="Close" onClick=${onClose}>✕</button></div>
    <div class="preview">${preview && canPreview ? (
        info.kind === "pdf" ? html`<iframe src=${raw}></iframe>` :
        info.kind === "image" ? html`<img src=${raw} />` :
        info.kind === "video" ? html`<video src=${raw} controls></video>` :
        info.kind === "audio" ? html`<audio src=${raw} controls></audio>` :
        html`<pre>${text == null ? "Loading…" : text}</pre>`)
      : html`<${Thumb} file=${info} big=${true} />`}</div>
    <div class="actions">
      <button class="btn primary small" onClick=${() => act("open")}>Open</button>
      <button class="btn small" onClick=${() => act("reveal")}>${config && config.platform === "win32" ? "Show in Explorer" : "Reveal in Finder"}</button>
      ${config && config.platform === "darwin" ? html`<button class="btn small" onClick=${() => act("quicklook")}>Quick Look</button>` : null}
      ${canPreview ? html`<button class="btn small" onClick=${() => setPreview(!preview)}>${preview ? "Thumbnail" : "Preview"}</button>` : null}
      <button class="btn small" onClick=${() => onFocus(info.id)}>Show in graph</button>
    </div>
    ${info.offline ? html`<div class="note">☁️ This file is only in iCloud. Open it once in Finder to download it, then refresh the index to read its content.</div>` : null}
    <div class="block"><h5>Tags</h5>
      <div class="chips">${(info.tags || []).map((t) => html`<span class=${"chip" + (t.status === "suggested" ? " suggested" : "")} title=${(t.status === "suggested" ? "Suggested by the local model · " : "") + "source: " + (t.sources || []).join(", ")}>
          <${TagDot} color=${t.color} /><span onClick=${() => onView({ type: "tag", value: t.name, label: t.name })}>${t.name}</span>
          ${t.status === "suggested" ? html`<button class="x ok" title="Accept (also saves a Finder tag)" onClick=${() => tagAction(t.name, "accept")}>✓</button>
            <button class="x" title="Reject suggestion" onClick=${() => tagAction(t.name, "reject")}>✕</button>`
          : html`<button class="x" title=${(t.sources || []).includes("keyword") ? "Hide this keyword tag" : "Remove tag"} onClick=${() => tagAction(t.name, "remove")}>✕</button>`}
        </span>`)}
        ${!(info.tags || []).length ? html`<span class="note" style="padding:0">No tags yet.</span>` : null}</div>
      <form class="addtag" onSubmit=${(e) => { e.preventDefault(); if (newTag.trim()) { tagAction(newTag.trim(), "add"); setNewTag(""); } }}>
        <input placeholder="Add a tag…" value=${newTag} onInput=${(e) => setNewTag(e.target.value)} />
        <button class="btn small" type="submit">Add</button></form>
      ${config && config.finder_write ? html`<div class="note" style="padding:6px 0 0">Accepted and added tags are also saved as Finder tags.</div>` : null}
    </div>
    ${(info.topics || []).length ? html`<div class="block"><h5>Topic</h5><div class="chips">
      ${info.topics.map((t) => html`<span class="chip topic" onClick=${() => onView({ type: "topic", value: t.id, label: t.name })}>${t.name}</span>`)}</div></div>` : null}
    ${(info.concepts || []).length ? html`<div class="block"><h5>Inside this file <span>${info.concept_count}</span></h5><div class="chips">
      ${info.concepts.map((c) => html`<span class="chip concept" onClick=${() => onFocus(c.id)}>${c.label}</span>`)}</div></div>` : null}
    ${(info.related || []).length ? html`<div class="block"><h5>Related files</h5><div class="rel">
      ${info.related.map((r) => html`<div class="item" onClick=${() => onSelect(r.id)}><span>${ICONS[r.kind] || "📦"}</span>
        <span class="name">${r.name}<div class="why">${r.why}</div></span></div>`)}</div></div>` : null}
    <div class="block"><h5>Details</h5><dl class="kv">
      <dt>Folder</dt><dd><a onClick=${() => onView({ type: "folder", value: parentPath(info.path), label: parentPath(info.path) || "Top folder" })}>${parentPath(info.path) || "/"}</a></dd>
      <dt>Kind</dt><dd>${KIND_LABEL[info.kind] || info.kind}</dd>
      ${info.size != null ? html`<dt>Size</dt><dd>${fmtSize(info.size)}</dd>` : null}
      ${info.modified ? html`<dt>Modified</dt><dd>${fmtDate(info.modified)}</dd>` : null}
      ${info.created ? html`<dt>Created</dt><dd>${fmtDate(info.created)}</dd>` : null}
      ${info.title ? html`<dt>Title</dt><dd>${info.title}</dd>` : null}
      ${info.author ? html`<dt>Author</dt><dd>${info.author}</dd>` : null}
      ${info.pages ? html`<dt>Pages</dt><dd>${info.pages}</dd>` : null}
    </dl></div>
  </div>`;
}

// ---------------------------------------------------------------- chat
function renderAnswer(text, pathToId, onSelect) {
  const parts = [];
  const re = /`([^`]+)`|\*\*([^*]+)\*\*/g;
  let last = 0; let m; let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      const id = pathToId[m[1]] || pathToId[m[1].split("/").pop()];
      parts.push(id ? html`<a class="file" key=${k++} onClick=${() => onSelect(id)}>${m[1]}</a>` : html`<code key=${k++}>${m[1]}</code>`);
    } else parts.push(html`<b key=${k++}>${m[2]}</b>`);
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

const TOOL_LABEL = { search: "Searching", find_files: "Filtering files", get_file: "Reading file details",
  related_files: "Looking for related files", read_file: "Reading", list_tags: "Listing tags", list_topics: "Listing topics",
  list_folder: "Opening folder", explain: "Exploring", connection: "Tracing connection", overview: "Getting an overview" };

function Chat({ config, onSelect, onHighlight, onClose }) {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef(null);
  const [briefs, setBriefs] = useState({});
  useEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight; }, [msgs]);

  const send = async (question) => {
    question = (question || input).trim();
    if (!question || busy) return;
    setInput("");
    const history = msgs.filter((m) => m.role === "user" || (m.role === "assistant" && m.text)).map((m) => ({ role: m.role, content: m.text }));
    const bot = { role: "assistant", text: "", steps: [], files: [], pending: true };
    let next = [...msgs, { role: "user", text: question }, bot];
    setMsgs(next); setBusy(true);
    const update = (fn) => { fn(bot); next = [...next.slice(0, -1), { ...bot }]; setMsgs(next); };
    try {
      const res = await fetch("/api/chat", { method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, history }) });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, idx); buf = buf.slice(idx + 2);
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "token") update((b) => { b.text += ev.text; });
          else if (ev.type === "tool_call") update((b) => { b.steps.push({ name: ev.name, args: ev.args, done: false }); });
          else if (ev.type === "tool_result") {
            update((b) => { const s = b.steps[b.steps.length - 1]; if (s) { s.done = true; s.summary = ev.summary; } });
            if (ev.files && ev.files.length) onHighlight(ev.files);
          } else if (ev.type === "done") {
            update((b) => { b.text = ev.answer || b.text; b.files = ev.files || []; b.pending = false; });
            loadBriefs(ev.files || []);
          } else if (ev.type === "error") update((b) => { b.error = ev.message; b.pending = false; });
        }
      }
    } catch (e) {
      update((b) => { b.error = e.message; b.pending = false; });
    } finally { setBusy(false); }
  };
  const loadBriefs = async (ids) => {
    const missing = ids.filter((id) => !briefs[id]).slice(0, 12);
    const got = {};
    await Promise.all(missing.map((id) => api(`/api/file?${qs({ id })}`).then((f) => { got[id] = f; }).catch(() => {})));
    setBriefs((b) => ({ ...b, ...got }));
  };
  const pathToId = useMemo(() => {
    const m = {};
    Object.values(briefs).forEach((f) => { m[f.path] = f.id; m[f.name] = f.id; });
    msgs.forEach((x) => (x.files || []).forEach((id) => { const p = id.replace(/^file:/, ""); m[p] = id; m[p.split("/").pop()] = id; }));
    return m;
  }, [briefs, msgs]);

  const ready = config && config.chat_ready;
  const examples = ["What is this collection about?", "Which slides talk about our roadmap?", "Show me the most recent PDFs", "Find everything related to the budget"];
  return html`<div class="chat">
    <div class="chat-head"><b>Ask your files</b><span class="pill">${config ? config.chat_model : ""}</span>
      <button class="btn ghost small" onClick=${onClose}>✕</button></div>
    ${config && !config.ollama ? html`<div class="warnbox">The local model is not running. Install <b>Ollama</b> from ollama.com, open it, then run <code>graphify fs setup</code> to download the models.</div>`
      : config && !ready ? html`<div class="warnbox">Model <code>${config.chat_model}</code> is not downloaded yet. Run <code>graphify fs setup</code> (or <code>ollama pull ${config.chat_model}</code>).</div>` : null}
    <div class="msgs" ref=${box}>
      ${!msgs.length ? html`<div class="suggestions"><div class="note" style="padding:0 0 4px">The local model searches the graph for you. Try:</div>
        ${examples.map((q) => html`<button onClick=${() => send(q)} disabled=${!ready}>${q}</button>`)}</div>` : null}
      ${msgs.map((m) => m.role === "user" ? html`<div class="msg user">${m.text}</div>` : html`<div class="msg bot">
        ${m.steps.length ? html`<div class="steps">${m.steps.map((s) => html`<div class="step">${s.done ? "✓" : html`<span class="spin"></span>`}
          ${TOOL_LABEL[s.name] || s.name}${s.args && (s.args.query || s.args.tag || s.args.name || s.args.file) ? `: ${s.args.query || s.args.tag || s.args.name || s.args.file}` : ""}${s.summary ? ` → ${s.summary}` : ""}</div>`)}</div>` : null}
        ${m.pending && !m.text ? html`<div class="step"><span class="spin"></span> Thinking…</div>` : null}
        <div class="text">${renderAnswer(m.text, pathToId, onSelect)}</div>
        ${m.error ? html`<div class="warnbox">${m.error}</div>` : null}
        ${!m.pending && m.files.length ? html`<div class="files">${m.files.slice(0, 8).map((id) => {
          const f = briefs[id]; const name = f ? f.name : id.replace(/^file:/, "").split("/").pop();
          return html`<div class="item" onClick=${() => onSelect(id)}><span>${ICONS[f ? f.kind : "other"]}</span><span class="name">${name}</span>
            <span class="count">${f ? parentPath(f.path) : ""}</span></div>`; })}</div>` : null}
      </div>`)}
    </div>
    <form class="composer" onSubmit=${(e) => { e.preventDefault(); send(); }}>
      <textarea placeholder=${ready ? "Ask about your files… (Enter to send)" : "Chat needs the local model"} value=${input} disabled=${!ready}
        onInput=${(e) => setInput(e.target.value)} onKeyDown=${(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}></textarea>
      <button class="btn primary" type="submit" disabled=${busy || !ready}>${busy ? "…" : "Send"}</button>
    </form>
  </div>`;
}

// ---------------------------------------------------------------- app
function App() {
  const [config, setConfig] = useState(null);
  const [facets, setFacets] = useState(null);
  const [view, setView] = useState({ type: "recent", label: "Recent files" });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [tab, setTab] = useState("files");
  const [layout, setLayout] = useState(localStorage.getItem("gfs-layout") || "grid");
  const [graph, setGraph] = useState(null);
  const [focus, setFocus] = useState(null);
  const [chatOpen, setChatOpen] = useState(localStorage.getItem("gfs-chat") === "1");
  const [query, setQuery] = useState("");
  const [folders, setFolders] = useState({});
  const [toastMsg, setToastMsg] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const searchRef = useRef(null);

  const toast = (m) => { setToastMsg(m); clearTimeout(toast.t); toast.t = setTimeout(() => setToastMsg(null), 3500); };
  const loadFacets = () => api("/api/facets").then(setFacets).catch((e) => toast(e.message));
  useEffect(() => {
    api("/api/config").then(setConfig).catch((e) => toast(e.message));
    loadFacets();
    const onKey = (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "k") { e.preventDefault(); searchRef.current && searchRef.current.focus(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => localStorage.setItem("gfs-layout", layout), [layout]);
  useEffect(() => localStorage.setItem("gfs-chat", chatOpen ? "1" : "0"), [chatOpen]);

  const loadFolder = (path) => api(`/api/folder?${qs({ path })}`).then((d) => setFolders((f) => ({ ...f, [path]: d })));

  // Load the file list for the current view.
  useEffect(() => {
    setLoading(true);
    let p;
    if (view.type === "search") p = api(`/api/search?${qs({ q: view.value, limit: 60 })}`);
    else if (view.type === "folder") p = api(`/api/folder?${qs({ path: view.value })}`).then((d) => ({ files: d.files, folders: d.folders, total: d.total_files, folder: d }));
    else {
      const params = { recent: {}, tag: { tag: view.value }, topic: { topic: view.value }, kind: { kind: view.value }, year: { year: view.value }, highlight: {} }[view.type] || {};
      if (view.type === "highlight") p = Promise.all(view.value.slice(0, 60).map((id) => api(`/api/file?${qs({ id })}`).catch(() => null)))
        .then((fs) => ({ files: fs.filter(Boolean).map((f) => ({ ...f, tags: (f.tags || []).filter((t) => t.status === "accepted").map((t) => t.name) })), total: fs.filter(Boolean).length }));
      else p = api(`/api/files?${qs({ ...params, limit: 300 })}`);
    }
    p.then((r) => setResult(r)).catch((e) => { toast(e.message); setResult({ files: [] }); }).finally(() => setLoading(false));
  }, [view]);

  // Graph data follows focus (or the current view when nothing is focused).
  useEffect(() => {
    if (tab !== "graph" && tab !== "map") return;
    let target = focus;
    if (tab === "map") target = null;
    else if (!target) {
      if (view.type === "tag" && facets) { const t = facets.tags.find((x) => x.name === view.value); target = t ? t.id : null; }
      else if (view.type === "topic") target = `topic:${view.value}`;
      else if (view.type === "folder") target = `dir:${view.value}`;
      else if (selected) target = selected;
    }
    const p = target ? api(`/api/neighborhood?${qs({ id: target })}`) : api("/api/topicmap");
    p.then(setGraph).catch((e) => toast(e.message));
  }, [tab, focus, view, selected, facets]);

  const selectFile = (id) => { setSelected(id); };
  const openFile = (id) => post("/api/open", { id, action: "open" }).catch((e) => toast(e.message));
  const go = (v) => { setView(v); setFocus(null); setTab("files"); };
  const doSearch = (e) => { e && e.preventDefault(); if (query.trim()) go({ type: "search", value: query.trim(), label: `“${query.trim()}”` }); };
  const focusNode = (id) => { setFocus(id); setTab("graph"); };
  const onGraphNode = (id, n) => {
    if (!n) return;
    if (n.fs_kind === "file") setSelected(id);
  };
  const onGraphDbl = (id, n) => {
    if (!n) return;
    if (n.fs_kind === "file") { setSelected(id); setFocus(id); }
    else if (n.group === "topic" && tab === "map") { setFocus(id); setTab("graph"); }
    else setFocus(id);
  };
  const refresh = async () => {
    setRefreshing(true);
    try {
      await post("/api/refresh", {});
      for (let i = 0; i < 600; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        const s = await api("/api/refresh");
        if (!s.running) break;
      }
      await loadFacets(); setFolders({}); setView({ ...view }); toast("Index refreshed");
    } catch (e) { toast(e.message); } finally { setRefreshing(false); }
  };

  const files = (result && result.files) || [];
  const total = result ? (result.total != null ? result.total : files.length) : 0;
  const cls = "app" + (selected ? "" : " no-inspector") + (chatOpen ? " chat-open" : "");
  const crumbs = view.type === "folder" ? (view.value ? view.value.split("/") : []) : null;

  return html`<div class=${cls}>
    <header class="top">
      <div class="brand"><img src="/static/icon.svg" /><span>graphify fs</span><small>${config ? config.root_name : ""}</small></div>
      <form class="searchbox" onSubmit=${doSearch}>🔎<input ref=${searchRef} placeholder="Search names, tags and what's inside your files…" value=${query} onInput=${(e) => setQuery(e.target.value)} /><kbd>⌘K</kbd></form>
      <div class="spacer"></div>
      ${config ? html`<span class=${"pill " + (config.chat_ready ? "ok" : "warn")} title=${config.ollama ? "Ollama is running" : "Ollama not running"}>${config.chat_ready ? "● local model ready" : "○ local model off"}</span>` : null}
      <button class="btn" onClick=${refresh} disabled=${refreshing} title="Rescan folders, files and tags (fast)">${refreshing ? "Refreshing…" : "↻ Refresh"}</button>
      <button class=${"btn" + (chatOpen ? " primary" : "")} onClick=${() => setChatOpen(!chatOpen)}>💬 Ask</button>
    </header>

    <nav class="side">
      <div class="sec">
        <div class=${"item" + (view.type === "recent" ? " active" : "")} onClick=${() => go({ type: "recent", label: "Recent files" })}><span>🕘</span><span class="name">Recent files</span><span class="count">${facets ? facets.total_files + "" : ""}</span></div>
        <div class=${"item" + (view.type === "folder" && view.value === "" ? " active" : "")} onClick=${() => go({ type: "folder", value: "", label: "Top folder" })}><span>🏠</span><span class="name">Top folder</span></div>
        <div class=${"item" + (tab === "map" ? " active" : "")} onClick=${() => { setFocus(null); setTab("map"); }}><span>🗺️</span><span class="name">Topic map</span></div>
      </div>
      <div class="sec"><h4>Folders</h4>
        <${FolderTree} path="" depth=${0} current=${view.type === "folder" ? view.value : null} cache=${folders} load=${loadFolder}
          onPick=${(p) => go({ type: "folder", value: p, label: p.split("/").pop() })} /></div>
      ${facets ? html`
        <${Section} title="Tags" items=${facets.tags} filterable=${true} limit=${14} render=${(t) => html`<div class=${"item" + (view.type === "tag" && view.value === t.name ? " active" : "")}
            onClick=${() => go({ type: "tag", value: t.name, label: `# ${t.name}` })} title=${"sources: " + t.sources.join(", ")}>
            <${TagDot} color=${t.color} /><span class="name">${t.name}</span><span class="count">${t.files}</span></div>`} />
        <${Section} title="Topics" items=${facets.topics} filterable=${true} limit=${10} render=${(t) => html`<div class=${"item" + (view.type === "topic" && view.value === t.id ? " active" : "")}
            onClick=${() => go({ type: "topic", value: t.id, label: t.name })} title=${(t.top_tags || []).join(", ")}>
            <span>🧭</span><span class="name">${t.name}</span><span class="count">${t.files}</span></div>`} />
        <${Section} title="Kinds" items=${facets.kinds} limit=${14} render=${(k) => html`<div class=${"item" + (view.type === "kind" && view.value === k.id ? " active" : "")}
            onClick=${() => go({ type: "kind", value: k.id, label: k.label })}><span>${ICONS[k.id] || "📦"}</span><span class="name">${k.label}</span><span class="count">${k.files}</span></div>`} />
        <${Section} title="Years" items=${facets.years.map((y) => ({ ...y, name: y.year }))} limit=${8} render=${(y) => html`<div class=${"item" + (view.type === "year" && view.value === y.year ? " active" : "")}
            onClick=${() => go({ type: "year", value: y.year, label: `Modified in ${y.year}` })}><span>📅</span><span class="name">${y.year}</span><span class="count">${y.files}</span></div>`} />
      ` : null}
    </nav>

    <main class="main">
      <div class="toolbar">
        <h2>${tab === "map" ? "Topic map" : tab === "graph" && graph && graph.center
          ? "Around: " + ((graph.nodes.find((n) => n.id === graph.center) || {}).label || graph.center) : view.label}</h2>
        <span class="sub">${tab === "files" ? (loading ? "loading…" : `${total} file${total === 1 ? "" : "s"}`) : ""}</span>
        <div class="spacer" style="flex:1"></div>
        ${tab === "files" ? html`<div class="tabs"><button class=${layout === "grid" ? "on" : ""} onClick=${() => setLayout("grid")}>▦</button><button class=${layout === "list" ? "on" : ""} onClick=${() => setLayout("list")}>☰</button></div>` : null}
        <div class="tabs">
          <button class=${tab === "files" ? "on" : ""} onClick=${() => setTab("files")}>Files</button>
          <button class=${tab === "graph" ? "on" : ""} onClick=${() => setTab("graph")}>Graph</button>
          <button class=${tab === "map" ? "on" : ""} onClick=${() => { setFocus(null); setTab("map"); }}>Map</button>
        </div>
      </div>
      ${tab === "files" ? html`<div class="content">
        ${crumbs ? html`<div class="crumbs"><a onClick=${() => go({ type: "folder", value: "", label: "Top folder" })}>🏠</a>
          ${crumbs.map((c, i) => html`<span>/</span><a onClick=${() => go({ type: "folder", value: crumbs.slice(0, i + 1).join("/"), label: c })}>${c}</a>`)}</div>` : null}
        ${view.type === "search" && result && ((result.concepts || []).length || (result.tags || []).length) ? html`<div class="hits">
          ${(result.tags || []).length ? html`<span class="label">Tags</span>${result.tags.map((t) => html`<span class="chip" onClick=${() => go({ type: "tag", value: t.name, label: `# ${t.name}` })}># ${t.name} · ${t.files}</span>`)}` : null}
          ${(result.concepts || []).length ? html`<span class="label">Concepts</span>${result.concepts.map((c) => html`<span class="chip concept" title=${c.topic || ""} onClick=${() => focusNode(c.id)}>${c.label} · ${c.files}</span>`)}` : null}
        </div>` : null}
        ${result && result.folders && result.folders.length ? html`<div class="grid" style="margin-bottom:12px">${result.folders.map((d) => html`<div class="card folder" onClick=${() => go({ type: "folder", value: d.path, label: d.name })}>
          <div class="thumb"><span class="ico">📁</span></div><div class="meta"><div class="fname">${d.name}</div><div class="fsub">${d.files} files</div></div></div>`)}</div>` : null}
        ${!loading && !files.length && !(result && result.folders && result.folders.length) ? html`<div class="empty"><h3>Nothing here</h3>${view.type === "search" ? "Try other words, or ask the local model in the chat." : "No files match this view."}</div>` : null}
        ${layout === "grid" ? html`<div class="grid">${files.map((f) => html`<${FileCard} f=${f} selected=${f.id === selected} onSelect=${selectFile} onOpen=${openFile} />`)}</div>`
          : files.length ? html`<${FileTable} files=${files} selected=${selected} onSelect=${selectFile} onOpen=${openFile} />` : null}
      </div>` : html`<${GraphView} data=${graph} focusId=${selected} onNode=${onGraphNode} onNodeDbl=${onGraphDbl} />`}
    </main>

    ${selected ? html`<${Inspector} fileId=${selected} config=${config} toast=${toast} onSelect=${selectFile} onFocus=${focusNode}
        onView=${go} onClose=${() => setSelected(null)} onTagsChanged=${loadFacets} />` : html`<div class="insp"></div>`}
    ${chatOpen ? html`<${Chat} config=${config} onSelect=${(id) => setSelected(id)} onClose=${() => setChatOpen(false)}
        onHighlight=${(ids) => { if (ids.length) setView({ type: "highlight", value: ids, label: "Files the assistant found" }); }} />` : null}
    ${toastMsg ? html`<div class="toast">${toastMsg}</div>` : null}
  </div>`;
}

render(html`<${App} />`, document.getElementById("app"));
