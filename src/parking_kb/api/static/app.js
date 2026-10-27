const form = document.getElementById("composer");
const input = document.getElementById("question");
const messages = document.getElementById("messages");
const docCount = document.getElementById("doc-count");

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function addMessage(role, text, abstained = false) {
  const section = document.createElement("section");
  section.className = `message ${role}${abstained ? " abstained" : ""}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(text).replaceAll("\n", "<br>");
  section.appendChild(bubble);
  messages.appendChild(section);
  messages.scrollTop = messages.scrollHeight;
  return section;
}

function renderCitations(citations) {
  const list = document.createElement("div");
  list.className = "citations";
  for (const citation of citations) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `[${citation.n}] 《${citation.title}》`;
    details.appendChild(summary);

    const meta = document.createElement("p");
    meta.className = "citation-meta";
    const parts = [citation.section_path, `p.${citation.page_start}`];
    if (citation.effective_date) {
      parts.push(`${citation.effective_date} 生效`);
    }
    meta.textContent = parts.filter(Boolean).join(" · ");
    details.appendChild(meta);

    const snippet = document.createElement("p");
    snippet.className = "citation-text";
    snippet.textContent = citation.text || "（原文片段不可用）";
    details.appendChild(snippet);
    list.appendChild(details);
  }
  return list;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage("user", question);
  input.value = "";
  const pending = addMessage("assistant", "检索并生成中…");
  pending.classList.add("pending");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    pending.remove();
    const bubble = addMessage("assistant", data.answer, data.abstained);
    if (data.citations && data.citations.length > 0) {
      bubble.querySelector(".bubble").appendChild(renderCitations(data.citations));
    }
  } catch (error) {
    pending.querySelector(".bubble").textContent =
      `服务暂不可用：${error.message}`;
    pending.classList.remove("pending");
  }
});

fetch("/documents")
  .then((response) => (response.ok ? response.json() : []))
  .then((documents) => {
    docCount.textContent = `已收录 ${documents.length} 份文档`;
  })
  .catch(() => {
    docCount.textContent = "知识库状态未知";
  });
