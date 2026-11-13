# graphify fs: a knowledge-graph explorer for your files

`graphify fs` turns a folder of everyday files (Word, PowerPoint, Excel, PDF, Markdown, images, iPhone photos, audio and video) into a knowledge graph. You can browse it in a local web app, with a **small local LLM that navigates the graph for you**, and you can expose it to Claude or any other MCP client.

Everything runs on your machine. Nothing is sent to a cloud model.

```
your folder ──► scan (folders, files, Finder tags, document keywords)        seconds
            ──► content (graphify extract + local model: concepts per file)  slow, cached
            ──► media (local Whisper transcripts of audio/video)             cached
            ──► topics (local model suggests topic tags, names the topics)   cached
            ──► ~/.graphify-fs/<folder>/graphify-out/fs/graph.json
                     ├─► graphify fs open      (web app + local chat; watches the folder)
                     ├─► Graph Files.app       (launcher; optional start at login)
                     └─► graphify fs mcp       (Claude Code / Claude Desktop)
```

## What you get

- **Folders, files and tags as a graph.** Folders `contain` files. Files are `tagged_with` tags: your Finder tags (with their colours), Office/PDF keywords, and topic tags the local model suggests. Tags that appear on the same files are linked by `co_occurs_with`, weighted by how often they share files.
- **What is inside each file.** graphify extracts concepts from each document, slide deck, PDF, image and transcript. Each file `mentions` its concepts, so you can go from a file to its concepts to every other file that discusses them.
- **Topics.** Communities of related concepts and files, named by the local model ("Atlas product roadmap", "2025 budget and invoices").
- **An explorer that replaces Finder for finding things:**
  - search across names, tags and content
  - filter by tag, topic, kind, folder or year
  - previews (PDF, images, HEIC, video, audio, text)
  - Open / Reveal in Finder / Quick Look
  - a graph view around any file, tag, concept or topic
  - a topic map
  - accepting or rejecting suggested tags
  - a chat panel where the local model answers questions about your files and links to them
- **Finder write-back.** Tags you accept or add in the explorer are also written as real Finder tags. Every write is logged and can be undone.
- **Always up to date.** While the explorer runs it watches the folder. New, edited, moved and deleted files appear within seconds, and the local model reads new contents in the background.
- **A real app.** `graphify fs app install` adds a **Graph Files** app to Spotlight, Launchpad and the Dock, and can start the explorer at login.

## 1. Install

From this repository (Python 3.11+, [uv](https://docs.astral.sh/uv/)):

```bash
uv tool install --python 3.12 ".[fs,video]"
```

Run it again after you pull changes. For development, `--editable` runs the code straight from the repository instead; if the repository is in iCloud Drive or Documents, the launcher app then needs macOS permission to read it (see [Launcher app](#6-launcher-app-and-start-at-login)).

## 2. Install the local models (one time)

1. Install **Ollama** from https://ollama.com/download and open it once (it runs in the menu bar).
2. Download the two models:

   ```bash
   graphify fs setup
   ```

| Role | Default | Why |
|---|---|---|
| Content extraction | `gemma3:12b` | Reads text **and images**, multilingual, about 8 GB. Good quality on a 24 GB Mac. |
| Chat and topic tags | `qwen3:8b` | Reliable tool calling, fast, about 5 GB. |
| Transcription | Whisper `small` | Runs on the CPU. Downloaded automatically the first time it is needed. |

On a smaller machine, use lighter models:

```bash
graphify fs setup --extract-model gemma3:4b --chat-model qwen3:4b
```

Then pass the same `--extract-model` and `--chat-model` to `graphify fs index`.

## 3. Index a folder

```bash
graphify fs index ~/Documents/Work
```

- **First run.** Folders, files and tags are ready in seconds. The content pass is the slow part: a local 12B model takes a few seconds per document, so a few thousand documents can take a night. Progress and an ETA are printed. You can stop it at any time; the next run resumes from the cache.
- **Later runs only re-read what changed.**
- **Fast refresh** (folders, files and tags only; reuses the cached content and topics; no LLM):

  ```bash
  graphify fs index ~/Documents/Work --metadata-only
  ```

  The explorer's **↻ Refresh** button does the same.
- **iCloud Drive "Optimize Mac Storage" files.** Files that are only in the cloud are listed (name, folder, tags) but **never downloaded**. Open them once in Finder, or set `GRAPHIFY_DOWNLOAD_ICLOUD=1`, to include their content.
- **Where the index lives.** The index is stored in `~/.graphify-fs/<folder>-<hash>/`. Nothing is written into your folder, except Finder tags you accept. Set `GRAPHIFY_FS_HOME` to store indexes elsewhere.

Useful flags: `--no-media` (skip audio/video), `--no-topics`, `--no-vision`, `--whisper-model medium`, `--topic-limit 200`.

## 4. Explore

```bash
graphify fs open ~/Documents/Work
```

This starts the explorer in the background (if it is not running yet) and opens a browser tab on `http://127.0.0.1:8765/?t=<token>`. The explorer keeps running after you close the tab; `graphify fs stop` ends it. `graphify fs explore` does the same in the foreground of your terminal (Ctrl-C stops it).

- **⌘K** to search. Results come from file names, titles, tags, keywords and the concepts extracted from contents.
- The **sidebar** has Recent files, Folders, Tags (with Finder colours), Topics, Kinds and Years.
- **Click** a file to open the inspector (preview, tags, topic, what's inside, related files and why they are related). **Double-click** opens the file in its app.
- The **Graph** tab shows the neighbourhood of the selected file, tag, concept or topic. Double-click a node to explore around it.
- The **Map** tab shows topics and how they connect.
- In the inspector, **Tags** shows suggested tags with a dashed border. **✓** accepts a tag (and writes it to Finder), **✕** rejects it so it is never suggested again, and **Add a tag…** creates your own.
- **💬 Ask** opens the local chat. Ask things like "which slides talk about the roadmap?", "what did we decide about the GPU budget?" or "show me last month's invoices". The model calls the graph tools itself, the steps appear as it works, and the files it cites are clickable.

Undo the last Finder tag write(s):

```bash
graphify fs undo-tags -n 1
```

To browse without ever writing Finder tags:

```bash
graphify fs explore --no-finder-write
```

## 5. Keep it up to date

While the explorer runs, it watches the indexed folder (FSEvents on macOS):

- **Within seconds** of a change (a few seconds after the changes stop), names, folders, Finder tags, keywords and the full-text index are refreshed. New files are searchable by the words inside them right away, and deleted or moved files disappear. Open views refresh by themselves.
- **Contents** of new and edited files are read by the local model (concepts, transcripts, topic tags) once the folder has been calm for 5 minutes, in the background, and only while Ollama runs with the models. Results are cached per file, so this only reads what changed.

The header shows where things stand:

| Pill | Meaning |
|---|---|
| ✓ Up to date | watching; nothing waiting |
| ↻ Changes detected… / Updating… | a quick refresh is coming or running |
| 📖 N to read | N new or edited files have not been read by the local model yet. Click it to read them now. Hover it to see why it is waiting (for example "local model not running"). |
| 📖 Reading files… | the local model is reading them |

Options for `open`, `explore` and `serve`: `--no-watch` (never follow changes) and `--no-content-updates` (follow changes, but never start the local model in the background). Without the explorer, `graphify fs index ~/Documents/Work --watch` indexes, then keeps watching in the terminal.

Only one index run happens at a time per folder. A long content pass started from the terminal does not block the quick refreshes, and the next one waits for it to finish.

## 6. Launcher app and start at login

On macOS, give it a proper app:

```bash
graphify fs app install ~/Documents/Work --login
```

- **Graph Files** appears in `~/Applications`, Spotlight and Launchpad. Drag it to the Dock to keep it there. Opening it does what `graphify fs open` does.
- `--login` adds a login item, so the explorer starts in the background when you log in and keeps the index up to date even with no window open. It also restarts the explorer within 15 minutes if it ever stops.
- **Privacy.** macOS protects iCloud Drive, Documents, Desktop, Downloads and external drives per app. The first time Graph Files opens one of them, macOS asks to let **Graph Files** access it: click **Allow**. The login item starts the explorer through the app, so this one permission covers both. The install command tells you which places need it.
- Several folders: install once per folder. The second app is named "Graph Files - <folder>".
- `graphify fs status` shows whether the explorer is running and starts at login. `graphify fs app uninstall ~/Documents/Work` removes the app and the login item.

## 7. Use it from Claude (MCP)

The same tools the local chat uses are available over MCP:

| Tool | What it does |
|---|---|
| `overview` | counts by kind, top tags, top topics |
| `search` | free text over names, tags and content |
| `find_files` | tag / kind / folder / topic / name / date filters |
| `get_file` | metadata, tags, topic, concepts, related files |
| `related_files` | ranked by shared concepts, tags, topic and folder |
| `list_tags`, `list_topics`, `list_folder` | browse the collection |
| `explain` | a concept, tag or topic and the files that mention it |
| `connection` | the shortest chain linking two files, tags or concepts |

**Claude Code:**

```bash
claude mcp add graphify-fs -- graphify fs mcp ~/Documents/Work
```

**Claude Desktop:** add this to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "graphify-fs": { "command": "graphify", "args": ["fs", "mcp", "/Users/you/Documents/Work"] }
  }
}
```

The server reloads the graph automatically after a re-index.

## 8. Ask from the terminal

```bash
graphify fs ask "what is this folder about?"
graphify fs ask "find the latest budget spreadsheet" ~/Documents/Work --json
```

## Privacy and safety

- Content extraction, transcription, tagging and chat all run locally through Ollama and faster-whisper.
- The explorer is locked down:
  - it listens on `127.0.0.1` only;
  - every API call needs a random per-launch token (an HttpOnly, SameSite=Strict cookie);
  - the `Host` header is checked to block DNS rebinding;
  - cross-origin POSTs are refused;
  - previews, thumbnails and Open/Reveal only work for files in the index, under the indexed folder.
- Finder tags are written only when you click ✓ or Add, only to files under the indexed folder, and every change is logged in `tag-writes.jsonl` for `undo-tags`.

## Supported files

| Kind | Content extraction |
|---|---|
| Word `.docx`, PowerPoint `.pptx` (slides + speaker notes), Excel `.xlsx` | converted to markdown, then concepts |
| PDF | text, then concepts; PDF keywords become tags |
| Markdown, text, HTML | concepts |
| Images `.png .jpg .webp .gif`, iPhone `.heic` | vision model (HEIC converted with `sips`) |
| Audio and video `.mp3 .m4a .wav .mp4 .mov …` | Whisper transcript, then concepts |
| Everything else (`.key`, `.pages`, `.zip`, apps …) | listed with folder, dates, size and Finder tags |

## Graph format

`fs/graph.json` is a regular graphify graph (`graphify query`, `graphify path` and `graphify explain` work on it), plus:

| Node id | Meaning | Main attributes |
|---|---|---|
| `dir:<path>` | folder (`dir:` = the root) | `fs_kind: folder`, `file_count`, `size`, `finder_tags` |
| `file:<path>` | file | `fs_kind: file`, `kind`, `ext`, `size`, `modified`, `created`, `offline`, `finder_tags`, `keywords`, `title`, `author`, `pages`, `tags`, `topics` |
| `tag:<key>` | tag | `fs_kind: tag`, `tag_sources` (finder/keyword/llm/user), `color`, `file_count`, `topics` |
| other ids | concepts from graphify's content graph | `community`, `community_name` |

| Edge | From → to | Notes |
|---|---|---|
| `contains` | folder → folder / file | |
| `mentions` | file → concept | from the concept's (or its edges') `source_file` |
| `tagged_with` | file / folder → tag | `sources`, `status` (`accepted` / `suggested`), confidence `EXTRACTED` (human) or `INFERRED` (LLM) |
| `co_occurs_with` | tag ↔ tag | `weight` = Jaccard overlap of their files, `shared_files` |

## Troubleshooting

- **The explorer says "local model off".** Open Ollama, then run `graphify fs setup`.
- **The content pass is slow.** Use smaller models, `--no-vision`, or index a sub-folder first. Results are cached per file, so an interrupted run loses nothing.
- **A file shows "iCloud".** It isn't downloaded. Open it in Finder; the explorer picks it up by itself (or click **↻ Refresh**).
- **Graph Files does nothing when opened.** macOS is waiting for a privacy permission; after 20 seconds the app shows a notification saying so. Look for a dialog, or allow **Graph Files** in System Settings > Privacy & Security > Files and Folders. `graphify fs open` from Terminal always works (it uses Terminal's permission). If graphify was installed with `--editable` from a folder in iCloud Drive or Documents, reinstall it without `--editable`.
- **"📖 N to read" never goes away.** Hover it: the local model is not running or not downloaded. Open Ollama and run `graphify fs setup`.
- **Wrong tag suggestions.** Click ✕ once; rejected tags are never suggested again for that file.
