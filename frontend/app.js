"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  resume: null,
  jd: null,
  resumeName: "",
  jdName: "",
  difficulty: "中等",
  sessionId: null,
  eventSource: null,
  status: "ready",
  question: null,
  report: null,
  history: [],
  answers: [],
  processing: false,
  timerId: null,
  elapsedSeconds: 0,
  statusEl: null,
  lastQuestionKey: "",
};

const ANALYZE_STEPS = [
  "已接收简历与 JD",
  "正在分析简历与 JD",
  "正在规划面试",
  "正在构建知识库检索",
  "面试准备就绪",
];

const statusPill = $("status-pill");
const statusText = $("status-text");

function setStatus(text, tone) {
  statusText.textContent = text;
  statusPill.classList.remove("ready", "busy", "done", "error");
  if (tone) {
    statusPill.classList.add(tone);
  }
}

function setView(viewId) {
  ["setup-view", "analyzing-view", "interview-view", "report-view", "history-view"].forEach((id) => {
    $(id).classList.toggle("active", id === viewId);
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(seconds) {
  const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
  const secs = String(seconds % 60).padStart(2, "0");
  return `${minutes}:${secs}`;
}

function bindUpload(inputId, panelId, metaId, btnId, key) {
  const input = $(inputId);
  const panel = $(panelId);
  const meta = $(metaId);
  const btn = $(btnId);

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) return;
    if (!/\.(md|txt|docx|pdf)$/i.test(file.name)) {
      input.value = "";
      showToast("暂不支持该文件格式");
      return;
    }
    state[key] = file;
    meta.textContent = `${file.name} · ${formatSize(file.size)}`;
    btn.textContent = "替换文件";
    panel.classList.add("ready");
    updateStart();
  });
}

function resetUpload(inputId, panelId, metaId, btnId, key) {
  const input = $(inputId);
  input.value = "";
  state[key] = null;
  $(panelId).classList.remove("ready");
  $(metaId).textContent = "未上传";
  $(btnId).textContent = "选择文件";
}

function updateStart() {
  $("start-btn").disabled = !(state.resume && state.jd);
}

function markStep(index, status) {
  const step = document.querySelectorAll("#step-list .step")[index];
  if (!step) return;
  step.classList.remove("pending", "active", "done", "error");
  step.classList.add(status);
}

function resetSteps() {
  document.querySelectorAll("#step-list .step").forEach((_, index) => {
    markStep(index, "pending");
  });
}

function updateProgressFill(stepIndex) {
  $("progress-fill").style.width = `${((stepIndex + 1) / ANALYZE_STEPS.length) * 100}%`;
}

function finishAnalysis() {
  stopTimer();
  ANALYZE_STEPS.forEach((_, index) => markStep(index, "done"));
  $("progress-fill").style.width = "100%";
}

function startTimer() {
  stopTimer();
  state.elapsedSeconds = 0;
  $("elapsed").textContent = "00:00";
  state.timerId = setInterval(() => {
    state.elapsedSeconds += 1;
    $("elapsed").textContent = formatTime(state.elapsedSeconds);
  }, 1000);
}

function stopTimer() {
  if (state.timerId) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
}

async function readError(response) {
  try {
    const data = await response.json();
    if (data && data.detail) return String(data.detail);
    if (data && data.message) return String(data.message);
  } catch (error) {
    // fall through to status text
  }
  return `请求失败 (${response.status})`;
}

async function startAnalysis() {
  if (!state.resume || !state.jd) return;

  const form = new FormData();
  form.append("resume_file", state.resume);
  form.append("jd_file", state.jd);
  form.append("difficulty", state.difficulty);

  setView("analyzing-view");
  setStatus("分析中", "busy");
  $("end-btn").classList.remove("hidden");
  $("history-return-btn").classList.add("hidden");
  resetSteps();
  markStep(0, "done");
  markStep(1, "active");
  updateProgressFill(1);
  state.resumeName = state.resume.name;
  state.jdName = state.jd.name;
  $("side-resume").textContent = state.resume.name;
  $("side-jd").textContent = state.jd.name;
  $("side-role").textContent = "解析中";
  $("side-topics").textContent = "规划中";
  $("analyze-error").classList.add("hidden");
  startTimer();

  try {
    const response = await fetch("/api/sessions", { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    state.sessionId = data.session_id;
    state.lastQuestionKey = "";
    await syncSession();
    connectEvents();
  } catch (error) {
    stopTimer();
    showAnalyzeError(error.message);
  }
}

async function syncSession() {
  const response = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}`);
  if (!response.ok) throw new Error(await readError(response));
  applySessionState(await response.json());
}

function applySessionState(data) {
  state.status = data.status;
  state.history = data.history || [];

  if (data.question) {
    state.question = data.question;
    state.lastQuestionKey = questionKey(data.question);
    renderHistory(state.history);
    renderQuestion(data.question);
    finishAnalysis();
    setView("interview-view");
    setStatus("面试中", "ready");
    state.processing = false;
  }

  if (data.report) {
    state.report = data.report;
    renderReport(data.report);
    setView("report-view");
    setStatus("面试完成", "done");
  }

  if (data.error && !data.question) {
    showAnalyzeError(data.error);
  }
}

function connectEvents() {
  closeEvents();
  const source = new EventSource(`/api/sessions/${encodeURIComponent(state.sessionId)}/events`);
  state.eventSource = source;
  source.addEventListener("status", (event) => handleStatus(JSON.parse(event.data)));
  source.addEventListener("question", (event) => handleQuestion(JSON.parse(event.data)));
  source.addEventListener("report", (event) => handleReport(JSON.parse(event.data)));
  source.addEventListener("error", (event) => handleError(JSON.parse(event.data)));
}

function closeEvents() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function handleStatus(data) {
  state.status = data.status;
  if (data.status === "analyzing") {
    markStep(0, "done");
    markStep(1, "active");
    setStatus("分析中", "busy");
    updateProgressFill(1);
  } else if (data.status === "planning") {
    markStep(1, "done");
    markStep(2, "active");
    setStatus("规划中", "busy");
    updateProgressFill(2);
  } else if (data.status === "preparing_knowledge") {
    markStep(2, "done");
    markStep(3, "active");
    setStatus("准备中", "busy");
    updateProgressFill(3);
  } else if (data.status === "reporting") {
    setStatus("生成报告", "busy");
  }
}

function questionKey(question) {
  return `${state.sessionId}:${question.question_index}:${question.question}`;
}

function handleQuestion(data) {
  const key = questionKey(data);
  const isNew = key !== state.lastQuestionKey;
  state.lastQuestionKey = key;
  state.question = data;
  state.processing = false;
  finishAnalysis();
  renderQuestion(data);
  setView("interview-view");
  setStatus("面试中", "ready");

  if (isNew) {
    removeSystemStatus();
    appendInterviewerBubble(data.question, data.topic);
    state.history.push({ role: "interviewer", content: data.question });
    scrollChat();
  }
  setComposerEnabled(true);
}

function handleReport(data) {
  state.report = data;
  state.processing = false;
  removeSystemStatus();
  renderReport(data);
  saveHistory(data);
  setView("report-view");
  setStatus("面试完成", "done");
}

function handleError(data) {
  const message = data.message || "面试处理失败";
  state.processing = false;
  removeSystemStatus();
  if ($("analyzing-view").classList.contains("active")) {
    stopTimer();
    showAnalyzeError(message);
  } else {
    setComposerEnabled(true);
    showToast(message);
  }
}

function showAnalyzeError(message) {
  $("analyze-error-text").textContent = message;
  $("analyze-error").classList.remove("hidden");
  setStatus("分析失败", "error");
}

function renderHistory(history) {
  const body = $("chat-body");
  body.innerHTML = "";
  (history || []).forEach((message) => {
    if (message.role === "interviewer") {
      appendInterviewerBubble(message.content, "");
    } else {
      appendUserBubble(message.content);
    }
  });
}

function appendInterviewerBubble(text, topic) {
  const bubble = document.createElement("div");
  bubble.className = "msg interviewer";
  const topicEl = document.createElement("span");
  topicEl.className = "q-topic";
  topicEl.textContent = topic || "";
  const textEl = document.createElement("span");
  textEl.textContent = text;
  bubble.append(topicEl, textEl);
  $("chat-body").appendChild(bubble);
}

function appendUserBubble(text) {
  const bubble = document.createElement("div");
  bubble.className = "msg user";
  bubble.textContent = text;
  $("chat-body").appendChild(bubble);
}

function addSystemStatus(text, withSpinner) {
  removeSystemStatus();
  const item = document.createElement("div");
  item.className = "msg system";
  if (withSpinner) {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    const label = document.createElement("span");
    label.textContent = text;
    item.append(spinner, label);
  } else {
    item.textContent = text;
  }
  $("chat-body").appendChild(item);
  scrollChat();
  state.statusEl = item;
  return item;
}

function removeSystemStatus() {
  if (state.statusEl) {
    state.statusEl.remove();
    state.statusEl = null;
  }
}

function scrollChat() {
  const body = $("chat-body");
  body.scrollTop = body.scrollHeight;
}

function renderQuestion(data) {
  renderTopics(data.topics || [], data.topic_index || 0);
  renderCandidate(data.candidate || {});
  updateProgress(data);
}

function renderTopics(topics, activeIndex) {
  const list = $("topic-list");
  list.innerHTML = "";
  topics.forEach((topic, index) => {
    const item = document.createElement("li");
    item.className = "topic-item";
    if (index === activeIndex) item.classList.add("active");
    else if (index < activeIndex) item.classList.add("done");
    const check = document.createElement("span");
    check.className = "topic-check";
    check.textContent = index + 1;
    const title = document.createElement("span");
    title.textContent = topic.title || "面试提问";
    item.append(check, title);
    list.appendChild(item);
  });
}

function renderCandidate(candidate) {
  $("candidate-name").textContent = candidate.name || "候选人";
  $("candidate-role").textContent = candidate.target_position || "待定";
  $("candidate-company").textContent = candidate.company || "待定";
}

function updateProgress(data) {
  const current = Number(data.question_index) || 1;
  const total = Number(data.total_questions) || 15;
  $("current-num").textContent = current;
  $("chat-meta").textContent = `第 ${current} 题 · ${data.topic}`;
  $("progress-sub").textContent = `已完成 ${Math.max(current - 1, 0)} 题`;
  $("question-progress").style.width = `${Math.min(((current - 1) / total) * 100, 100)}%`;
}

function setComposerBusy(busy) {
  const input = $("answer-input");
  const btn = $("send-btn");
  input.disabled = busy;
  btn.disabled = busy;
  btn.textContent = busy ? "评分中..." : "发送";
}

function setComposerEnabled(enabled) {
  const input = $("answer-input");
  const btn = $("send-btn");
  input.disabled = !enabled;
  btn.disabled = !enabled || !input.value.trim();
  btn.textContent = "发送";
  if (enabled) {
    input.focus();
  }
}

async function sendAnswer() {
  const input = $("answer-input");
  const text = input.value.trim();
  if (!text || state.processing || !state.sessionId) return;

  state.processing = true;
  appendUserBubble(text);
  state.history.push({ role: "user", content: text });
  input.value = "";
  setComposerBusy(true);
  addSystemStatus("正在评分与准备下一题...", true);

  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer: text }),
    });
    if (!response.ok) throw new Error(await readError(response));
    state.answers.push(text);
  } catch (error) {
    state.processing = false;
    removeSystemStatus();
    setComposerEnabled(true);
    showToast(error.message);
  }
}

function renderReport(report) {
  state.report = report;
  $("overall-score").textContent = Number(report.overall_score || 0).toFixed(1);
  $("report-session").textContent = report.session_id || state.sessionId || "-";
  $("summary-text").textContent = report.summary || "暂无整体评语";
  $("report-meta").textContent = `本次面试共 ${(report.rounds || []).length} 题`;

  const dimensionList = $("dimension-list");
  dimensionList.innerHTML = "";
  (report.dimension_scores || []).forEach((item) => {
    const score = Math.min(Math.max(Number(item.score || 0), 0), 10);
    const row = document.createElement("div");
    row.className = "dimension-row";
    const name = document.createElement("span");
    name.textContent = item.dimension || "维度";
    const barWrap = document.createElement("span");
    barWrap.className = "dim-bar";
    const fill = document.createElement("span");
    fill.className = "dim-fill";
    fill.style.width = `${score * 10}%`;
    barWrap.appendChild(fill);
    const scoreEl = document.createElement("span");
    scoreEl.className = "dim-score";
    scoreEl.textContent = score.toFixed(1);
    row.append(name, barWrap, scoreEl);
    dimensionList.appendChild(row);
  });

  const suggestionList = $("suggestion-list");
  suggestionList.innerHTML = "";
  (report.suggestions || []).forEach((item) => {
    const priorityText = { high: "高", medium: "中", low: "低" }[item.priority] || item.priority;
    const priorityClass = item.priority === "high" ? "high" : item.priority === "medium" ? "mid" : "low";
    const li = document.createElement("li");
    li.className = "suggestion-item";
    const priorityEl = document.createElement("span");
    priorityEl.className = `priority ${priorityClass}`;
    priorityEl.textContent = priorityText;
    const textEl = document.createElement("span");
    textEl.textContent = item.suggestion || item.dimension || "";
    li.append(priorityEl, textEl);
    suggestionList.appendChild(li);
  });
}

function downloadReport() {
  if (!state.report) {
    showToast("暂无报告");
    return;
  }
  const report = state.report;
  const suggestions = (report.suggestions || [])
    .map((item) => `- [${item.priority}] ${item.dimension || ""}: ${item.suggestion || ""}`)
    .join("\n");
  const lines = [
    "# 面试报告",
    "",
    `会话 ID: ${report.session_id || state.sessionId}`,
    `总分: ${Number(report.overall_score || 0).toFixed(1)} / 10`,
    "",
    report.summary || "",
    "",
    "## 改进建议",
    suggestions || "- 暂无",
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mock-interview-report.md";
  link.click();
  URL.revokeObjectURL(url);
  showToast("报告已下载");
}

// ====================== 历史面试（localStorage） ======================

const HISTORY_INDEX_KEY = "mi_history_index";
const HISTORY_RECORD_PREFIX = "mi_history_";

function saveHistory(report) {
  try {
    const sessionId = state.sessionId || report.session_id || ("session-" + Date.now());
    const record = {
      session_id: sessionId,
      created_at: new Date().toISOString(),
      difficulty: state.difficulty,
      resume_name: state.resumeName,
      jd_name: state.jdName,
      overall_score: Number(report.overall_score || 0),
      question_count: (report.rounds || []).length,
      report: report,
      history: state.history || [],
    };
    // 存完整记录
    localStorage.setItem(HISTORY_RECORD_PREFIX + sessionId, JSON.stringify(record));
    // 更新索引（摘要列表，按时间倒序）
    const index = loadHistoryIndex();
    const summary = {
      session_id: sessionId,
      created_at: record.created_at,
      difficulty: record.difficulty,
      resume_name: record.resume_name,
      jd_name: record.jd_name,
      overall_score: record.overall_score,
      question_count: record.question_count,
    };
    // 去重：同 session_id 覆盖
    const filtered = index.filter((item) => item.session_id !== sessionId);
    filtered.unshift(summary);
    localStorage.setItem(HISTORY_INDEX_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.warn("保存历史面试失败:", error);
  }
}

function loadHistoryIndex() {
  try {
    const raw = localStorage.getItem(HISTORY_INDEX_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function loadHistoryRecord(sessionId) {
  try {
    const raw = localStorage.getItem(HISTORY_RECORD_PREFIX + sessionId);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function deleteHistory(sessionId) {
  try {
    localStorage.removeItem(HISTORY_RECORD_PREFIX + sessionId);
    const index = loadHistoryIndex().filter((item) => item.session_id !== sessionId);
    localStorage.setItem(HISTORY_INDEX_KEY, JSON.stringify(index));
  } catch (error) {
    console.warn("删除历史记录失败:", error);
  }
}

function formatHistoryDate(iso) {
  try {
    const d = new Date(iso);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return iso;
  }
}

function renderHistoryList() {
  const list = loadHistoryIndex();
  const container = $("history-list");
  if (!list.length) {
    container.innerHTML = '<div class="history-empty">暂无历史记录</div>';
    return;
  }
  container.innerHTML = "";
  list.forEach((item) => {
    const card = document.createElement("div");
    card.className = "history-card";
    card.innerHTML = `
      <div class="history-card-main">
        <div class="history-card-title">${formatHistoryDate(item.created_at)}</div>
        <div class="history-card-meta">
          <span class="history-tag">${item.difficulty || "中等"}</span>
          <span>${item.question_count || 0} 题</span>
          <span class="history-file" title="${item.resume_name || ""}">${item.resume_name || "-"}</span>
        </div>
      </div>
      <div class="history-card-right">
        <div class="history-card-score">
          <strong>${Number(item.overall_score || 0).toFixed(1)}</strong>
          <span>/ 10</span>
        </div>
        <button class="history-del-btn" type="button" title="删除">删除</button>
      </div>
    `;
    card.addEventListener("click", () => viewHistory(item.session_id));
    card.querySelector(".history-del-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      if (window.confirm("确定删除这条历史记录吗？")) {
        deleteHistory(item.session_id);
        renderHistoryList();
      }
    });
    container.appendChild(card);
  });
}

function viewHistory(sessionId) {
  const record = loadHistoryRecord(sessionId);
  if (!record) {
    showToast("历史记录不存在或已损坏");
    return;
  }
  // 渲染对话历史
  renderHistory(record.history || []);
  // 渲染报告
  if (record.report) {
    renderReport(record.report);
  }
  // 禁用 composer，显示查看模式
  setComposerEnabled(false);
  $("send-btn").textContent = "历史记录";
  $("end-btn").classList.add("hidden");
  $("history-return-btn").classList.remove("hidden");
  setView("interview-view");
  setStatus("查看历史", "ready");
}

function resetAll() {
  closeEvents();
  stopTimer();
  state.sessionId = null;
  state.question = null;
  state.report = null;
  state.history = [];
  state.answers = [];
  state.processing = false;
  state.statusEl = null;
  state.lastQuestionKey = "";
  resetUpload("resume-input", "resume-panel", "resume-meta", "resume-btn", "resume");
  resetUpload("jd-input", "jd-panel", "jd-meta", "jd-btn", "jd");
  const defaultDiff = document.querySelector('input[name="difficulty"][value="中等"]');
  if (defaultDiff) defaultDiff.checked = true;
  state.difficulty = "中等";
  $("chat-body").innerHTML = "";
  $("progress-fill").style.width = "0%";
  $("question-progress").style.width = "0%";
  setView("setup-view");
  updateStart();
  setStatus("准备就绪", "ready");
}

let toastTimer = null;
function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2400);
}

bindUpload("resume-input", "resume-panel", "resume-meta", "resume-btn", "resume");
bindUpload("jd-input", "jd-panel", "jd-meta", "jd-btn", "jd");

document.querySelectorAll('input[name="difficulty"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    if (radio.checked) state.difficulty = radio.value;
  });
});

$("start-btn").addEventListener("click", startAnalysis);
$("retry-btn").addEventListener("click", () => {
  closeEvents();
  stopTimer();
  setView("setup-view");
  setStatus("准备就绪", "ready");
});
$("send-btn").addEventListener("click", sendAnswer);
$("answer-input").addEventListener("input", () => {
  if (!state.processing) {
    $("send-btn").disabled = !$("answer-input").value.trim();
  }
});
$("answer-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendAnswer();
  }
});
$("end-btn").addEventListener("click", async () => {
  if (!state.sessionId) return;
  if (!window.confirm("确定结束面试吗？")) return;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}`, {
      method: "DELETE",
    });
    if (!response.ok && response.status !== 404) {
      throw new Error(await readError(response));
    }
    showToast("面试已结束");
    resetAll();
  } catch (error) {
    showToast(error.message);
  }
});
$("restart-btn").addEventListener("click", resetAll);
$("download-btn").addEventListener("click", downloadReport);

$("history-btn").addEventListener("click", () => {
  renderHistoryList();
  setView("history-view");
});
$("history-back").addEventListener("click", () => {
  setView("report-view");
});
$("history-return-btn").addEventListener("click", () => {
  renderHistoryList();
  setView("history-view");
});

updateStart();
setStatus("准备就绪", "ready");
