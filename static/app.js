/* ============================================
   RAG Embedding Comparison — app.js
   Chat-style UI with Tabs & Override support
   ============================================ */

// ===== State =====
let currentSessionId = null;
let selectedFiles = [];
const STEP_ORDER = ['upload', 'chunk', 'embed', 'rag', 'evaluate', 'wer'];
let pipelineState = {};
STEP_ORDER.forEach(s => pipelineState[s] = 'idle');

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
  loadSessions();
  setupUploadZone();
  initQuestions();
  checkPromptStatus();
  loadLLMConfig();
});

// ===== LLM Parameters =====
function collectLLMParams() {
  return {
    temperature: parseFloat(document.getElementById('llmTemperature').value),
    top_p:       parseFloat(document.getElementById('llmTopP').value),
    max_predict: parseInt(document.getElementById('llmMaxPredict').value),
    num_ctx:     parseInt(document.getElementById('llmNumCtx').value) || 0,
  };
}

async function loadLLMConfig() {
  try {
    const res = await fetch('/api/llm-config');
    const d = await res.json();
    document.getElementById('llmTemperature').value = d.temperature;
    document.getElementById('llmTopP').value        = d.top_p;
    document.getElementById('llmMaxPredict').value  = d.max_predict;
    document.getElementById('llmNumCtx').value      = d.num_ctx;
  } catch (e) { /* keep HTML defaults */ }
}

// ===== Tab Switching =====
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-content').forEach(c => {
    c.style.display = 'none';
    c.classList.remove('active');
  });
  const el = document.getElementById(`tab-${tab}`);
  if (el) { el.style.display = 'block'; el.classList.add('active'); }
}

// ===== Golden Modal =====
async function openGoldenModal() {
  document.getElementById('goldenModal').classList.remove('hidden');
  // If a session is active, try to load its saved questions
  if (currentSessionId) {
    try {
      const res = await fetch(`/api/questions/${currentSessionId}`);
      const data = await res.json();
      if (data.status === 'success' && data.questions.length > 0) {
        const container = document.getElementById('questionsContainer');
        container.innerHTML = '';
        questionIdCounter = 0;
        data.questions.forEach(q => addQuestionBlock(q.question, q.answer));
        return;
      }
    } catch (e) { /* keep current state */ }
  }
}
function closeGoldenModal() { document.getElementById('goldenModal').classList.add('hidden'); }

// ===== Sessions =====
async function loadSessions() {
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    const sel = document.getElementById('sessionSelect');
    sel.innerHTML = '<option value="">-- เลือก Session --</option>';
    (data.sessions || []).forEach(s => {
      sel.innerHTML += `<option value="${s.id}">[${s.id}] ${s.filename} — ${s.status}</option>`;
    });
  } catch (e) { log('❌ Error loading sessions'); }
}

function onSessionChange(val) {
  currentSessionId = val ? parseInt(val) : null;
  const badge = document.getElementById('sessionBadge');
  if (currentSessionId) {
    badge.textContent = `Session #${currentSessionId}`;
    badge.className = 'px-2.5 py-1 rounded-full text-[10px] font-semibold bg-indigo-50 text-indigo-600 border border-indigo-200';
    syncTimelineWithSession();
  } else {
    badge.textContent = 'No Session';
    badge.className = 'px-2.5 py-1 rounded-full text-[10px] font-semibold bg-gray-100 text-gray-500 border border-gray-200';
    resetTimeline();
  }
}

// ===== Upload Zone =====
function setupUploadZone() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('fileInput');
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    selectedFiles = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
    showSelectedFiles();
  });
  input.addEventListener('change', () => { selectedFiles = Array.from(input.files); showSelectedFiles(); });
}

function showSelectedFiles() {
  const list = document.getElementById('fileList');
  list.innerHTML = selectedFiles.map(f => `<div class="text-[10px] text-gray-500">📎 ${f.name} (${(f.size / 1024).toFixed(0)} KB)</div>`).join('');
}

// ===== Default Q&A Dataset =====
const DEFAULT_QUESTIONS = [
  {
    q: "ประกาศ ธปท. เรื่อง หลักเกณฑ์การบริหารจัดการภัยทุจริตดิจิทัล (Digital Fraud Management) ฉบับนี้ มีผลบังคับใช้ตั้งแต่วันที่เท่าใดเป็นต้นไป และมีข้อกำหนดใดที่ได้รับการยกเว้นให้ขยายระยะเวลาบังคับใช้ออกไป?",
    a: "ประกาศมีผลบังคับใช้ตั้งแต่วันที่ 17 ธันวาคม 2568 เป็นต้นไป ยกเว้นข้อ 5.3.1 (เรื่องการกำหนดนโยบายและการกำกับดูแลการบริหารจัดการภัยทุจริตดิจิทัล) ที่ให้ใช้บังคับเมื่อพ้นกำหนด 90 วันนับแต่วันถัดจากวันประกาศในราชกิจจานุเบกษา"
  },
  {
    q: "ตามหลักเกณฑ์นี้ หน่วยงานใดบ้างภายในองค์กร (Three Lines of Defense) ที่ถูกระบุไว้อย่างชัดเจนว่าต้องมีส่วนร่วมในการผลักดันให้องค์กรมีกระบวนการควบคุมดูแลความเสี่ยงที่ดี เพื่อจัดการภัยทุจริตดิจิทัลได้อย่างเหมาะสมและทันกาล?",
    a: "หน่วยงานที่ทำหน้าที่บริหารความเสี่ยง หน่วยงานที่ทำหน้าที่กำกับการปฏิบัติตามกฎเกณฑ์ และหน่วยงานที่ทำหน้าที่ตรวจสอบภายใน"
  },
  {
    q: "ในกระบวนการรู้จักลูกค้า (KYC) และตรวจสอบเพื่อทราบข้อเท็จจริง (CDD) หากพบว่าลูกค้ามีความเสี่ยงต่อการนำบัญชีไปใช้เป็น \"บัญชีม้า\" ผู้ให้บริการทางการเงินจะต้องดำเนินการยกระดับการตรวจสอบอย่างไร และต้องตรวจสอบข้อมูลใดเพิ่มเติมบ้าง?",
    a: "ต้องยกระดับการตรวจสอบเพื่อทราบข้อเท็จจริงเกี่ยวกับลูกค้าให้อยู่ในระดับที่เข้มข้น (Enhanced Customer Due Diligence: EDD) โดยต้องตรวจสอบข้อมูลเพิ่มเติม เช่น แหล่งที่มาของเงินหรือทรัพย์สิน แหล่งที่มาของฐานะความมั่งคั่ง ข้อมูลเกี่ยวกับการประกอบกิจการ อาชีพ ชื่อและสถานที่ตั้งของที่ทำงาน"
  },
  {
    q: "เพื่อให้การติดตามและตรวจจับความผิดปกติในการทำธุรกรรมทางการเงินในเชิงรุก (Proactive detection) มีประสิทธิภาพและเท่าทันรูปแบบภัยทุจริตใหม่ๆ ประกาศฉบับนี้แนะนำให้นำเทคโนโลยีใดมาประยุกต์ใช้?",
    a: "การนำระบบวิเคราะห์ข้อมูล (data analytics) หรือเทคโนโลยีปัญญาประดิษฐ์ (artificial intelligence) มาใช้"
  },
  {
    q: "ในด้านการจัดการภัยทุจริตดิจิทัล (Actions) ผู้ให้บริการทางการเงินสามารถใช้มาตรการใดได้บ้างในการจัดการกับบัญชีที่อาจเข้าข่ายเป็น \"บัญชีม้า\" เพื่อจำกัดและระงับความเสียหาย?",
    a: "สามารถดำเนินการระงับเงินเข้า ระงับเงินออก ระงับการให้บริการผ่านช่องทางอิเล็กทรอนิกส์ ปฏิเสธการเปิดบัญชีใหม่ และการควบคุมการถอนเงินสด"
  },
  {
    q: "สำหรับช่องทางในการรับแจ้งเหตุการณ์ต้องสงสัยหรือเหตุการณ์ภัยทุจริตดิจิทัล (Hotline) ธปท. กำหนดมาตรฐานขั้นต่ำในการให้บริการไว้อย่างไร เพื่อให้ลูกค้าสามารถแจ้งเหตุได้อย่างสะดวกและรวดเร็ว?",
    a: "ต้องจัดให้มีช่องทางติดต่อเร่งด่วน (hotline) ทางโทรศัพท์ หรือทางอิเล็กทรอนิกส์ ที่เพียงพอและให้บริการอย่างต่อเนื่องทั้งในและนอกเวลาทำการ (24x7)"
  },
  {
    q: "ในกระบวนการแก้ไขสถานการณ์และการดูแลลูกค้า หากพิสูจน์ได้ว่าลูกค้าได้รับความเสียหายจากภัยทุจริตดิจิทัลอันเกิดจาก \"ความบกพร่องของผู้ให้บริการทางการเงิน\" ผู้ให้บริการทางการเงินมีหน้าที่ต้องดำเนินการอย่างไร?",
    a: "ผู้ให้บริการทางการเงินต้องทำการเยียวยาความเสียหายให้แก่ลูกค้าอย่างรวดเร็ว"
  },
  {
    q: "เมื่อเกิดเหตุการณ์ภัยทุจริตดิจิทัลในระดับที่มีความรุนแรง ส่งผลให้เกิดความเสียหายกับลูกค้าในวงกว้าง หรือส่งผลกระทบต่อชื่อเสียงขององค์กร ผู้ให้บริการทางการเงินมีหน้าที่ต้องรายงานเหตุการณ์ดังกล่าวให้หน่วยงานใดทราบโดยเร็ว?",
    a: "ต้องรายงานให้ \"ธนาคารแห่งประเทศไทย\" ทราบโดยเร็วตามช่องทางที่กำหนด"
  },
  {
    q: "นอกเหนือจากการสร้างความตระหนักรู้เชิงรุกแล้ว ธนาคารแห่งประเทศไทยสนับสนุนให้ผู้ให้บริการทางการเงินประเมินความตระหนักรู้ต่อภัยทุจริตดิจิทัล (awareness test) ของลูกค้า เพื่อนำผลประเมินไปใช้ประโยชน์ในการป้องกันภัยทุจริตอย่างไรบ้าง?",
    a: "สามารถนำผลการประเมินไปใช้เป็นปัจจัยเพิ่มเติมในการกำหนดระดับความเสี่ยงของลูกค้า กำหนดความเข้มข้นในการสร้างความตระหนักรู้ หรือใช้กำหนดค่าเริ่มต้นของวงเงินการทำธุรกรรมต่อวันของลูกค้าได้"
  },
  {
    q: "หากผู้ให้บริการทางการเงิน หรือกรรมการ ถูกเปรียบเทียบปรับหรือถูกกล่าวโทษจากการฝ่าฝืนประกาศฉบับนี้ ผู้ให้บริการทางการเงินมีหน้าที่ต้องเปิดเผยข้อมูลดังกล่าวตามหลักเกณฑ์ใด เว้นแต่ธนาคารแห่งประเทศไทยจะแจ้งไม่ให้เปิดเผย?",
    a: "ต้องเปิดเผยข้อมูลตามหลักเกณฑ์การเปิดเผยข้อมูลที่กำหนดในประกาศธนาคารแห่งประเทศไทย ว่าด้วยการบริหารจัดการด้านการให้บริการแก่ลูกค้าอย่างเป็นธรรม (Market Conduct) โดยอนุโลม"
  },
];

// ===== Questions (Golden Dataset) =====
let questionIdCounter = 0;

function initQuestions() {
  const container = document.getElementById('questionsContainer');
  container.innerHTML = '';
  questionIdCounter = 0;
  DEFAULT_QUESTIONS.forEach(item => addQuestionBlock(item.q, item.a));
}

function addQuestionBlock(q = '', a = '') {
  const uid = ++questionIdCounter;
  const container = document.getElementById('questionsContainer');
  const div = document.createElement('div');
  div.className = 'question-block relative';
  div.dataset.uid = uid;
  div.innerHTML = `
    <div class="flex items-center justify-between mb-1">
      <span class="q-num-label text-[11px] font-bold text-gray-400"></span>
      <button onclick="removeQuestionBlock(this)"
        class="w-5 h-5 rounded hover:bg-red-50 flex items-center justify-center text-gray-300 hover:text-red-400 text-xs transition-all"
        title="ลบ">✕</button>
    </div>
    <label class="q-label">คำถาม</label>
    <textarea data-role="question" rows="2" placeholder="พิมพ์คำถาม..."></textarea>
    <label class="a-label">เฉลย</label>
    <textarea data-role="answer" rows="2" placeholder="พิมพ์เฉลย..."></textarea>
  `;
  div.querySelector('[data-role="question"]').value = q;
  div.querySelector('[data-role="answer"]').value = a;
  container.appendChild(div);
  updateQuestionNumbers();
  updateQuestionCount();
}

function removeQuestionBlock(btn) {
  btn.closest('.question-block').remove();
  updateQuestionNumbers();
  updateQuestionCount();
}

function updateQuestionNumbers() {
  document.querySelectorAll('#questionsContainer .question-block').forEach((b, i) => {
    const lbl = b.querySelector('.q-num-label');
    if (lbl) lbl.textContent = `ข้อที่ ${i + 1}`;
  });
}

function updateQuestionCount() {
  const count = document.querySelectorAll('#questionsContainer .question-block').length;
  const badge = document.getElementById('questionCountBadge');
  if (badge) badge.textContent = `${count} ข้อ`;
}

function addQuestion() {
  addQuestionBlock();
  document.querySelector('#questionsContainer .question-block:last-child')
    ?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function saveGoldenDataset() {
  if (!currentSessionId) {
    alert('กรุณาเลือก Session ก่อนบันทึก');
    return;
  }
  const questions = collectQuestions();
  if (questions.length === 0) {
    alert('ไม่มีคำถาม กรุณาเพิ่มคำถามก่อน');
    return;
  }
  const btn = document.getElementById('saveGoldenBtn');
  btn.disabled = true;
  btn.textContent = '⏳ กำลังบันทึก...';
  try {
    const res = await fetch(`/api/questions/${currentSessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ questions }),
    });
    const data = await res.json();
    if (data.status === 'success') {
      log(`✅ บันทึก Golden Dataset สำเร็จ (${data.count} ข้อ)`);
      btn.textContent = '✅ บันทึกแล้ว';
      setTimeout(() => { btn.textContent = '💾 บันทึก'; btn.disabled = false; }, 2000);
    } else {
      throw new Error(data.message);
    }
  } catch (e) {
    log(`❌ Error saving: ${e.message}`);
    btn.textContent = '💾 บันทึก';
    btn.disabled = false;
    alert(`บันทึกไม่สำเร็จ: ${e.message}`);
  }
}

function clearAllQuestions() {
  if (!confirm('ล้างคำถามทั้งหมดหรือไม่?')) return;
  document.getElementById('questionsContainer').innerHTML = '';
  questionIdCounter = 0;
  updateQuestionCount();
}

// ===== Evaluation Prompt Modal =====
async function openPromptModal() {
  document.getElementById('promptModal').classList.remove('hidden');
  const textarea = document.getElementById('promptTextarea');
  const badge = document.getElementById('promptModalBadge');
  textarea.value = '⏳ กำลังโหลด...';
  try {
    const res = await fetch('/api/prompt/evaluation');
    const data = await res.json();
    textarea.value = data.prompt;
    if (data.is_custom) {
      badge.textContent = 'กำหนดเอง';
      badge.className = 'px-2 py-0.5 rounded-full text-[10px] font-bold bg-orange-100 text-orange-600 border border-orange-200';
    } else {
      badge.textContent = 'ค่าเริ่มต้น';
      badge.className = 'px-2 py-0.5 rounded-full text-[10px] font-bold bg-gray-100 text-gray-500 border border-gray-200';
    }
  } catch (e) {
    textarea.value = '';
    log('❌ Error loading prompt: ' + e.message);
  }
}

function closePromptModal() {
  document.getElementById('promptModal').classList.add('hidden');
}

async function savePromptChanges() {
  const prompt = document.getElementById('promptTextarea').value;
  const btn = document.getElementById('savePromptBtn');
  btn.disabled = true;
  btn.textContent = '⏳ กำลังบันทึก...';
  try {
    const res = await fetch('/api/prompt/evaluation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.message);
    log('✅ บันทึก Evaluation Prompt สำเร็จ');
    btn.textContent = '✅ บันทึกแล้ว';
    // Update badges
    const badge = document.getElementById('promptModalBadge');
    badge.textContent = 'กำหนดเอง';
    badge.className = 'px-2 py-0.5 rounded-full text-[10px] font-bold bg-orange-100 text-orange-600 border border-orange-200';
    document.getElementById('promptCustomBadge').classList.remove('hidden');
    setTimeout(() => { btn.textContent = '💾 บันทึก'; btn.disabled = false; }, 2000);
  } catch (e) {
    log('❌ Error saving prompt: ' + e.message);
    alert('บันทึกไม่สำเร็จ: ' + e.message);
    btn.textContent = '💾 บันทึก';
    btn.disabled = false;
  }
}

async function resetPromptToDefault() {
  if (!confirm('รีเซ็ต prompt กลับเป็นค่าเริ่มต้นหรือไม่?')) return;
  try {
    const res = await fetch('/api/prompt/evaluation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reset: true }),
    });
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.message);
    log('✅ รีเซ็ต Evaluation Prompt เป็นค่าเริ่มต้นสำเร็จ');
    document.getElementById('promptCustomBadge').classList.add('hidden');
    // Reload prompt into textarea
    await openPromptModal();
  } catch (e) {
    log('❌ Error resetting prompt: ' + e.message);
    alert('รีเซ็ตไม่สำเร็จ: ' + e.message);
  }
}

// Check on load whether a custom prompt is active (show badge)
async function checkPromptStatus() {
  try {
    const res = await fetch('/api/prompt/evaluation');
    const data = await res.json();
    if (data.is_custom) {
      document.getElementById('promptCustomBadge').classList.remove('hidden');
    }
  } catch (e) { /* silent */ }
}

// ===== Logging =====
function log(msg) {
  const area = document.getElementById('logArea');
  const time = new Date().toLocaleTimeString('th-TH');
  const div = document.createElement('div');
  div.innerHTML = `<span class="text-gray-400">[${time}]</span> ${msg}`;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
  // Show badge if popup is closed
  const popup = document.getElementById('logPopup');
  if (popup && popup.classList.contains('hidden')) {
    const badge = document.getElementById('logBadge');
    if (badge) badge.classList.remove('hidden');
  }
}

// ===== Log Popup =====
function toggleLogPopup() {
  const popup = document.getElementById('logPopup');
  const badge = document.getElementById('logBadge');
  popup.classList.toggle('hidden');
  if (!popup.classList.contains('hidden')) {
    // Hide badge when opened
    if (badge) badge.classList.add('hidden');
    // Scroll to bottom
    const area = document.getElementById('logArea');
    area.scrollTop = area.scrollHeight;
  }
}

// ===== Timeline Control =====
function setStepState(step, state, statusText) {
  const el = document.getElementById(`step-${step}`);
  const statusEl = document.getElementById(`step-${step}-status`);
  if (!el) return;
  el.className = `tl-step ${state}`;
  pipelineState[step] = state;
  if (statusText && statusEl) statusEl.textContent = statusText;
}

function resetTimeline() {
  STEP_ORDER.forEach(s => setStepState(s, '', 'รอดำเนินการ'));
}

async function getSessionStatus(sessionId) {
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    const session = data.sessions.find(s => s.id === sessionId);
    return session ? session.status : null;
  } catch (e) { return null; }
}

function getCompletedSteps(status) {
  const map = {
    'ocr_done': ['upload'],
    'chunked': ['upload', 'chunk'],
    'embedded': ['upload', 'chunk', 'embed'],
    'rag_done': ['upload', 'chunk', 'embed', 'rag'],
    'evaluated': ['upload', 'chunk', 'embed', 'rag', 'evaluate'],
    'wer_done': ['upload', 'chunk', 'embed', 'rag', 'evaluate', 'wer'],
  };
  return map[status] || [];
}

async function syncTimelineWithSession() {
  if (!currentSessionId) return;
  const status = await getSessionStatus(currentSessionId);
  const completed = getCompletedSteps(status);
  resetTimeline();
  completed.forEach(s => setStepState(s, 'done', '✓ เสร็จแล้ว'));
  // Auto-load results if session has rag_done or further
  if (completed.includes('rag')) {
    await loadSessionResults(currentSessionId);
  }
}

// ===== Auto-load Results for Existing Session =====
async function loadSessionResults(sessionId) {
  try {
    log(`📥 [Session #${sessionId}] กำลังโหลดผลลัพธ์...`);
    const res = await fetch(`/api/results/${sessionId}`);
    const data = await res.json();
    const loaded = [];

    if (data.rag_results?.length > 0) {
      displayRAGResults(data.rag_results);
      loaded.push(`RAG ${data.rag_results.length} ข้อ (chunk_type: ${data.rag_results[0].chunk_type})`);
    }
    if (data.eval_summary?.length > 0) {
      displayEvalResults(data.eval_summary);
      const avg4b = (data.eval_summary.reduce((s, e) => s + (e.score_4b || 0), 0) / data.eval_summary.length).toFixed(1);
      const avg8b = (data.eval_summary.reduce((s, e) => s + (e.score_8b || 0), 0) / data.eval_summary.length).toFixed(1);
      loaded.push(`Eval avg: 4B=${avg4b} 8B=${avg8b}`);
    }
    if (data.wer_results?.length > 0) {
      displayWERResults(data.wer_results);
      loaded.push(`WER ${data.wer_results.length} หน้า`);
    }

    if (loaded.length > 0) {
      log(`✅ โหลดผลลัพธ์สำเร็จ — ${loaded.join(' | ')}`);
      switchTab('rag');
    } else {
      log('ℹ️ ยังไม่มีผลลัพธ์ใน Session นี้');
    }
  } catch (e) {
    log(`⚠️ โหลดผลลัพธ์ไม่สำเร็จ: ${e.message}`);
  }
}

// ===== Override Step =====
// Returns null = "off" (skip completed), or 1..6 = override starting from that step
function getOverrideFromStep() {
  const val = document.getElementById('overrideFromStep').value;
  return val === 'off' ? null : parseInt(val);
}

// ===== RUN FULL PIPELINE =====
async function runFullPipeline() {
  const btn = document.getElementById('runAllBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="flex items-center justify-center gap-2"><span class="spinner"></span> กำลังดำเนินการ...</span>';

  try {
    // Determine which steps to skip
    let skipSteps = [];
    const overrideFrom = getOverrideFromStep(); // null = off, 1-6 = start from step N

    if (overrideFrom === null && currentSessionId) {
      // Off: skip already-completed steps
      const status = await getSessionStatus(currentSessionId);
      skipSteps = getCompletedSteps(status);
      if (skipSteps.length > 0) {
        const labels = skipSteps.map(s => `${STEP_ORDER.indexOf(s) + 1}.${s}`).join(', ');
        log(`⏭️ Override ปิด — ข้ามขั้นที่ทำแล้ว: ${labels}`);
        skipSteps.forEach(s => setStepState(s, 'done', '✓ ข้ามแล้ว'));
      } else {
        log('▶️ Override ปิด — เริ่มจากต้น');
      }
    } else if (overrideFrom !== null) {
      // Override from step N: skip steps before N
      skipSteps = STEP_ORDER.slice(0, overrideFrom - 1);
      const stepName = ['Upload & OCR', 'Chunking', 'Embedding', 'RAG Query', 'Evaluation', 'WER'][overrideFrom - 1];
      if (skipSteps.length > 0) {
        log(`🔄 Override เริ่มจากขั้น ${overrideFrom} (${stepName}) — ข้ามขั้นก่อนหน้า: ${skipSteps.join(', ')}`);
        skipSteps.forEach(s => setStepState(s, 'done', '✓ ข้ามแล้ว'));
      } else {
        log(`🔄 Override ทุกขั้นตอน — เริ่มจากขั้น 1 (Upload & OCR)`);
      }
    }

    // ---------- Step 1: Upload & OCR ----------
    if (!skipSteps.includes('upload')) {
      if (selectedFiles.length === 0 && !currentSessionId) {
        log('❌ กรุณาเลือกไฟล์ PDF ก่อน');
        resetBtn(); return;
      }
      if (selectedFiles.length > 0) {
        setStepState('upload', 'active', 'กำลังอัปโหลด...');
        log('📄 [Step 1/6] กำลังอัปโหลดและ OCR...');

        const formData = new FormData();
        selectedFiles.forEach(f => formData.append('files', f));
        if (currentSessionId && overrideFrom !== null) {
          formData.append('override_session_id', currentSessionId);
        }

        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status !== 'success') throw new Error(data.message);

        currentSessionId = data.results[0].session_id;
        log(`✅ Upload สำเร็จ — Session #${currentSessionId} (${data.results[0].total_pages} หน้า)`);
        setStepState('upload', 'done', '✓ เสร็จแล้ว');
        await loadSessions();
        document.getElementById('sessionSelect').value = currentSessionId;
        onSessionChange(currentSessionId);
      } else {
        setStepState('upload', 'done', '✓ ใช้ session เดิม');
        log('⏭️ ไม่มีไฟล์ใหม่ — ใช้ session ปัจจุบัน');
      }
    }

    if (!currentSessionId) { log('❌ ไม่มี session'); resetBtn(); return; }

    // ---------- Step 2: Chunking ----------
    if (!skipSteps.includes('chunk')) {
      setStepState('chunk', 'active', 'กำลัง chunking...');
      log('📦 [Step 2/6] กำลังแบ่ง chunks (Recursive + Agentic)...');
      const res = await fetch(`/api/chunk/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ llm_params: collectLLMParams() }),
      });
      const data = await res.json();
      if (data.status !== 'success') throw new Error(data.message);
      log(`✅ Chunking สำเร็จ — Recursive: ${data.recursive_chunks} chunks, Agentic: ${data.agentic_chunks} chunks`);
      setStepState('chunk', 'done', '✓ เสร็จแล้ว');
    }

    // ---------- Step 3: Embedding ----------
    if (!skipSteps.includes('embed')) {
      setStepState('embed', 'active', 'กำลัง embed...');
      const chunkType = document.getElementById('chunkType').value;
      log(`🧮 [Step 3/6] กำลังสร้าง embeddings (${chunkType} chunks)...`);
      log('   🔵 กำลัง embed ด้วย qwen3-embedding:4B...');

      const res = await fetch(`/api/embed/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_type: chunkType }),
      });
      const data = await res.json();
      if (data.status !== 'success') throw new Error(data.message);
      log(`   🟣 Unload 4B → โหลด 8B...`);
      log(`✅ Embedding สำเร็จ — total: ${data.total_chunks} chunks | 4B: ${data.embeddings_4b} | 8B: ${data.embeddings_8b}`);
      setStepState('embed', 'done', '✓ เสร็จแล้ว');
    }

    // ---------- Step 3.5: Save Questions ----------
    const questions = collectQuestions();
    if (questions.length > 0) {
      log(`📝 บันทึก ${questions.length} คำถาม...`);
      const res = await fetch(`/api/questions/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ questions }),
      });
      const data = await res.json();
      if (data.status !== 'success') throw new Error(data.message);
      log(`✅ บันทึกคำถามสำเร็จ (${data.count} ข้อ)`);
    }

    // ---------- Step 4: RAG ----------
    if (!skipSteps.includes('rag')) {
      setStepState('rag', 'active', 'กำลัง RAG...');
      const chunkType = document.getElementById('chunkType').value;
      const topK = parseInt(document.getElementById('topKInput').value) || 5;
      log(`🚀 [Step 4/6] เริ่ม RAG pipeline (${chunkType} chunks, top_k=${topK}, 4B → 8B)...`);

      const res = await fetch(`/api/rag/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_type: chunkType, top_k: topK, llm_params: collectLLMParams() }),
      });
      const data = await res.json();
      if (data.status !== 'success') throw new Error(data.message);
      log(`✅ RAG สำเร็จ — ${data.results.length} คำถาม`);
      displayRAGResults(data.results);
      setStepState('rag', 'done', '✓ เสร็จแล้ว');
      switchTab('rag');
    }

    // ---------- Step 5: Evaluation ----------
    if (!skipSteps.includes('evaluate')) {
      setStepState('evaluate', 'active', 'กำลังประเมิน...');
      log('📊 [Step 5/6] กำลังประเมินผล...');
      const chunkTypeEval = document.getElementById('chunkType').value;
      const res = await fetch(`/api/evaluate/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_type: chunkTypeEval, llm_params: collectLLMParams() }),
      });
      const data = await res.json();
      if (data.status !== 'success') throw new Error(data.message);
      const scores = data.evaluations?.map(e => `${e.question_number}: 4B=${e.score_4b ?? '-'} 8B=${e.score_8b ?? '-'}`).join(', ') || '';
      log(`✅ Evaluation สำเร็จ (${data.evaluations?.length ?? 0} ข้อ)`);
      if (scores) log(`   📊 Scores — ${scores}`);
      displayEvalResults(data.evaluations);
      setStepState('evaluate', 'done', '✓ เสร็จแล้ว');
    }

    // ---------- Step 6: WER ----------
    if (!skipSteps.includes('wer')) {
      setStepState('wer', 'active', 'กำลังคำนวณ WER...');
      log('📏 [Step 6/6] กำลังคำนวณ WER...');
      const res = await fetch(`/api/wer/${currentSessionId}`, { method: 'POST' });
      const data = await res.json();
      if (data.status !== 'success') throw new Error(data.message);
      log('✅ WER คำนวณเสร็จ');
      displayWERResults(data.results || data);
      setStepState('wer', 'done', '✓ เสร็จแล้ว');
    }

    log('🎉 Pipeline เสร็จสมบูรณ์! — กำลัง refresh ผลลัพธ์ทุก tab...');
    await loadSessionResults(currentSessionId);

  } catch (err) {
    log(`❌ Error: ${err.message}`);
    const activeStep = STEP_ORDER.find(s => pipelineState[s] === 'active');
    if (activeStep) setStepState(activeStep, 'error', '❌ Error');
  }
  resetBtn();
}

function resetBtn() {
  const btn = document.getElementById('runAllBtn');
  btn.disabled = false;
  btn.innerHTML = '<span class="flex items-center justify-center gap-2 font-semibold">🚀 <span>Run Full Pipeline</span></span>';
}

function resetPipeline() {
  resetTimeline();
  selectedFiles = [];
  document.getElementById('fileList').innerHTML = '';
  document.getElementById('fileInput').value = '';
  document.getElementById('logArea').textContent = '';
  log('🔄 Pipeline reset — พร้อมใช้งาน');
  log('🔄 Pipeline reset');
}

// ===== Collect Questions =====
function collectQuestions() {
  const questions = [];
  document.querySelectorAll('#questionsContainer .question-block').forEach((block, i) => {
    const qEl = block.querySelector('[data-role="question"]');
    const aEl = block.querySelector('[data-role="answer"]');
    if (qEl && qEl.value.trim()) {
      questions.push({
        number: i + 1,
        question: qEl.value.trim(),
        answer: aEl ? aEl.value.trim() : '',
      });
    }
  });
  return questions;
}

// ===== Escape HTML =====
function escapeHtml(text) {
  if (!text) return '';
  const el = document.createElement('div');
  el.textContent = text;
  return el.innerHTML;
}

// ===== Display RAG Results =====
function displayRAGResults(results) {
  const container = document.getElementById('tab-rag');
  let html = '';
  results.forEach(r => {
    // Check if real similarity scores are available (not all zero)
    const hasSim4b = r.chunks_4b?.some(c => c.similarity > 0);
    const hasSim8b = r.chunks_8b?.some(c => c.similarity > 0);

    function buildChunksHtml(chunks, hasSim) {
      if (!chunks || chunks.length === 0) return '<span class="text-xs text-gray-600">ไม่มีข้อมูล</span>';
      return chunks.map((c, idx) => {
        const simBadge = hasSim
          ? (() => {
              const sim = (c.similarity * 100).toFixed(1);
              const color = c.similarity >= 0.8 ? '#22c55e' : c.similarity >= 0.6 ? '#f59e0b' : '#ef4444';
              return `<span class="shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded" style="background:${color}18;color:${color}">${sim}%</span>`;
            })()
          : `<span class="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-400">#${idx + 1}</span>`;
        const typeBadge = c.chunk_type === 'agentic'
          ? `<span class="shrink-0 text-[9px] px-1.5 py-0.5 rounded font-semibold bg-orange-100 text-orange-600">agentic</span>`
          : c.chunk_type === 'recursive'
            ? `<span class="shrink-0 text-[9px] px-1.5 py-0.5 rounded font-semibold bg-indigo-100 text-indigo-600">recursive</span>`
            : '';
        return `<div class="flex items-start gap-2 py-1.5 border-b border-gray-100 last:border-0">
          ${simBadge}${typeBadge}
          <span class="text-[11px] text-gray-500 leading-relaxed">${escapeHtml(c.text)}</span>
        </div>`;
      }).join('');
    }

    const avg4b = (hasSim4b && r.chunks_4b?.length)
      ? (r.chunks_4b.reduce((s, c) => s + c.similarity, 0) / r.chunks_4b.length * 100).toFixed(1)
      : null;
    const avg8b = (hasSim8b && r.chunks_8b?.length)
      ? (r.chunks_8b.reduce((s, c) => s + c.similarity, 0) / r.chunks_8b.length * 100).toFixed(1)
      : null;

    html += `
    <div class="result-card">
      <h3 class="text-sm font-semibold text-cyan-700 mb-1">ข้อที่ ${r.question_number}</h3>
      <p class="text-xs text-gray-500 mb-4">${escapeHtml(r.question_text)}</p>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
        <div class="model-answer m4b">
          <div class="label flex items-center justify-between">
            <span>🔵 Embedding 4B</span>
            ${avg4b != null ? `<span class="text-[10px] font-normal opacity-70">avg sim: ${avg4b}%</span>` : '<span class="text-[10px] font-normal opacity-40">sim: N/A</span>'}
          </div>
          <div class="text-gray-700 text-sm">${escapeHtml(r.answer_4b)}</div>
        </div>
        <div class="model-answer m8b">
          <div class="label flex items-center justify-between">
            <span>🟣 Embedding 8B</span>
            ${avg8b != null ? `<span class="text-[10px] font-normal opacity-70">avg sim: ${avg8b}%</span>` : '<span class="text-[10px] font-normal opacity-40">sim: N/A</span>'}
          </div>
          <div class="text-gray-700 text-sm">${escapeHtml(r.answer_8b)}</div>
        </div>
      </div>

      <div class="model-answer golden mb-3">
        <div class="label">✅ เฉลย (Golden Answer)</div>
        <div class="text-gray-700 text-sm">${escapeHtml(r.golden_answer)}</div>
      </div>

      <details class="mt-2">
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-700 transition-colors select-none">
          📎 ดู Retrieved Chunks พร้อม Similarity Scores
        </summary>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">
          <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
            <div class="text-[10px] font-bold text-indigo-600 uppercase mb-2">🔵 4B Chunks (${r.chunks_4b?.length || 0})</div>
            ${buildChunksHtml(r.chunks_4b, hasSim4b)}
          </div>
          <div class="bg-gray-50 rounded-lg p-3 border border-gray-100">
            <div class="text-[10px] font-bold text-purple-600 uppercase mb-2">🟣 8B Chunks (${r.chunks_8b?.length || 0})</div>
            ${buildChunksHtml(r.chunks_8b, hasSim8b)}
          </div>
        </div>
      </details>
    </div>`;
  });
  container.innerHTML = html || '<div class="text-center py-20 text-gray-600"><p class="text-sm">ยังไม่มีผลลัพธ์</p></div>';
}

// ===== Display Eval Results =====
function displayEvalResults(results) {
  const container = document.getElementById('tab-eval');
  if (!results || !Array.isArray(results) || results.length === 0) {
    container.innerHTML = '<div class="text-center py-20 text-gray-600"><div class="text-5xl mb-4 opacity-20">📝</div><p class="text-sm">ยังไม่มีผลการประเมิน</p></div>';
    return;
  }

  // Summary averages
  const v4 = results.filter(r => r.score_4b != null);
  const v8 = results.filter(r => r.score_8b != null);
  const avg4b = v4.length ? (v4.reduce((s, r) => s + r.score_4b, 0) / v4.length).toFixed(1) : '-';
  const avg8b = v8.length ? (v8.reduce((s, r) => s + r.score_8b, 0) / v8.length).toFixed(1) : '-';
  const win4b = results.filter(r => r.score_4b != null && r.score_8b != null && r.score_4b > r.score_8b).length;
  const win8b = results.filter(r => r.score_4b != null && r.score_8b != null && r.score_8b > r.score_4b).length;

  function scoreColor(s) {
    if (s == null) return '#94a3b8';
    return s >= 70 ? '#22c55e' : s >= 40 ? '#f59e0b' : '#ef4444';
  }
  function scoreLabel(s) {
    if (s == null) return '—';
    return s >= 70 ? 'ดีมาก' : s >= 40 ? 'พอใช้' : 'ต้องปรับปรุง';
  }

  let html = `
  <!-- Legend -->
  <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-4 text-xs text-blue-900">
    <div class="font-bold mb-2">📊 คะแนนการประเมิน (0–100 คะแนน) — ประเมินโดย LLM เปรียบเทียบกับ Golden Answer</div>
    <div class="flex flex-wrap gap-x-5 gap-y-1">
      <span><span class="font-bold text-green-600">70–100</span> = ดีมาก: คำตอบถูกต้อง ครบถ้วน ตรงประเด็น</span>
      <span><span class="font-bold text-yellow-600">40–69</span> = พอใช้: ถูกบางส่วน อาจขาดรายละเอียด</span>
      <span><span class="font-bold text-red-500">&lt;40</span> = ต้องปรับปรุง: ผิดหรือไม่ครบ</span>
    </div>
  </div>
  <!-- Summary stats -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
    <div class="stat-item">
      <div class="value" style="color:#6366f1">${avg4b}</div>
      <div class="stat-label">คะแนนเฉลี่ย 4B<br><span class="text-[9px]">(จาก 100 คะแนน)</span></div>
    </div>
    <div class="stat-item">
      <div class="value" style="color:#a855f7">${avg8b}</div>
      <div class="stat-label">คะแนนเฉลี่ย 8B<br><span class="text-[9px]">(จาก 100 คะแนน)</span></div>
    </div>
    <div class="stat-item">
      <div class="value" style="color:#6366f1">${win4b}</div>
      <div class="stat-label">ข้อที่ 4B ชนะ</div>
    </div>
    <div class="stat-item">
      <div class="value" style="color:#a855f7">${win8b}</div>
      <div class="stat-label">ข้อที่ 8B ชนะ</div>
    </div>
  </div>`;

  results.forEach(r => {
    const s4b = r.score_4b ?? null;
    const s8b = r.score_8b ?? null;
    const c4b = scoreColor(s4b), c8b = scoreColor(s8b);
    const l4b = scoreLabel(s4b), l8b = scoreLabel(s8b);
    const winner = (s4b != null && s8b != null)
      ? (s4b > s8b ? '🔵 4B ดีกว่า' : s4b < s8b ? '🟣 8B ดีกว่า' : '🟰 เท่ากัน') : '';

    const hasAnswers = r.answer_4b || r.answer_8b || r.golden_answer;

    html += `
    <div class="result-card">
      <div class="flex items-center justify-between mb-1">
        <h3 class="text-sm font-semibold text-gray-700">ข้อที่ ${r.question_number}</h3>
        ${winner ? `<span class="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-gray-100 text-gray-600">${winner}</span>` : ''}
      </div>
      <p class="text-xs text-gray-500 mb-3">${escapeHtml(r.question_text || '')}</p>

      <!-- Score boxes -->
      <div class="flex items-center gap-3 mb-4">
        <div class="score-box flex-col items-start" style="background:${c4b}18;color:${c4b}">
          <div>🔵 Embedding 4B</div>
          <div class="text-lg font-black mt-0.5">${s4b ?? '—'}<span class="text-xs font-normal opacity-60"> /100</span></div>
          <div class="text-[10px] font-semibold mt-0.5">${l4b}</div>
        </div>
        <div class="score-box flex-col items-start" style="background:${c8b}18;color:${c8b}">
          <div>🟣 Embedding 8B</div>
          <div class="text-lg font-black mt-0.5">${s8b ?? '—'}<span class="text-xs font-normal opacity-60"> /100</span></div>
          <div class="text-[10px] font-semibold mt-0.5">${l8b}</div>
        </div>
      </div>

      <!-- 3-column answer comparison -->
      ${hasAnswers ? `
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-3 mb-3">
        <div class="model-answer m4b">
          <div class="label">🔵 คำตอบ 4B</div>
          <div class="text-gray-700 text-xs leading-relaxed">${escapeHtml(r.answer_4b || '—')}</div>
        </div>
        <div class="model-answer m8b">
          <div class="label">🟣 คำตอบ 8B</div>
          <div class="text-gray-700 text-xs leading-relaxed">${escapeHtml(r.answer_8b || '—')}</div>
        </div>
        <div class="model-answer golden">
          <div class="label">✅ เฉลย (Golden)</div>
          <div class="text-gray-700 text-xs leading-relaxed">${escapeHtml(r.golden_answer || '—')}</div>
        </div>
      </div>` : ''}

      ${r.evaluation_text ? `
      <details class="mt-1">
        <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600 transition-colors select-none">
          💬 ดูการวิเคราะห์จาก LLM
        </summary>
        <div class="model-answer eval mt-2">
          <div class="text-gray-600 text-xs leading-relaxed whitespace-pre-wrap">${escapeHtml(r.evaluation_text)}</div>
        </div>
      </details>` : ''}

      ${(r.llm_prompt_4b || r.llm_prompt_8b) ? `
      <details class="mt-1">
        <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-600 transition-colors select-none">
          🔍 ดู Full Prompt ที่ใช้ Inference (แยก 4B / 8B)
        </summary>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-2">
          <div>
            <div class="text-[10px] font-bold text-indigo-600 uppercase mb-1">🔵 Full Prompt — Embedding 4B</div>
            <pre class="text-[10px] text-gray-600 bg-indigo-50 rounded-lg p-3 max-h-72 overflow-y-auto border border-indigo-100 whitespace-pre-wrap leading-relaxed">${escapeHtml(r.llm_prompt_4b || '(ไม่มีข้อมูล)')}</pre>
          </div>
          <div>
            <div class="text-[10px] font-bold text-purple-600 uppercase mb-1">🟣 Full Prompt — Embedding 8B</div>
            <pre class="text-[10px] text-gray-600 bg-purple-50 rounded-lg p-3 max-h-72 overflow-y-auto border border-purple-100 whitespace-pre-wrap leading-relaxed">${escapeHtml(r.llm_prompt_8b || '(ไม่มีข้อมูล)')}</pre>
          </div>
        </div>
      </details>` : ''}
    </div>`;
  });

  container.innerHTML = html;
}

// ===== Display WER Results =====
// Data format: [{page_number, wer_score, ocr_text, reference_text, image_url}]
// wer_score: 0.0 = perfect, 1.0 = all wrong, -1 = no reference file
function displayWERResults(results) {
  const container = document.getElementById('tab-wer');
  if (!results || !Array.isArray(results) || results.length === 0) {
    container.innerHTML = '<div class="text-center py-20 text-gray-600"><div class="text-5xl mb-4 opacity-20">📐</div><p class="text-sm">ยังไม่มีผล WER</p></div>';
    return;
  }

  const valid = results.filter(r => r.wer_score >= 0);
  const noRef  = results.filter(r => r.wer_score < 0);
  const avgWer = valid.length
    ? (valid.reduce((s, r) => s + r.wer_score, 0) / valid.length * 100).toFixed(1)
    : '-';

  function werLevel(score) {
    if (score <= 0.10) return { label: 'ดีมาก',        color: '#22c55e' };
    if (score <= 0.30) return { label: 'ดี',            color: '#84cc16' };
    if (score <= 0.50) return { label: 'พอใช้',         color: '#f59e0b' };
    return               { label: 'ต้องปรับปรุง',   color: '#ef4444' };
  }

  let html = `
  <!-- Legend -->
  <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-4 text-xs text-blue-900">
    <div class="font-bold mb-2">📐 WER — Word Error Rate (อัตราคำผิดพลาด)</div>
    <div class="text-blue-700 mb-1">เปรียบเทียบ OCR output กับ reference text (<code>best_ocr/page_N.txt</code>)</div>
    <div class="flex flex-wrap gap-x-5 gap-y-1 mt-1">
      <span><span class="font-bold text-green-600">0–10%</span> = ดีมาก</span>
      <span><span class="font-bold" style="color:#84cc16">10–30%</span> = ดี</span>
      <span><span class="font-bold text-yellow-600">30–50%</span> = พอใช้</span>
      <span><span class="font-bold text-red-500">&gt;50%</span> = ต้องปรับปรุง</span>
    </div>
  </div>
  <!-- Summary stats -->
  <div class="grid grid-cols-3 gap-3 mb-5">
    <div class="stat-item">
      <div class="value" style="color:#6366f1">${avgWer}%</div>
      <div class="stat-label">Avg WER<br><span class="text-[9px]">(เฉลี่ยทุกหน้า)</span></div>
    </div>
    <div class="stat-item">
      <div class="value" style="color:#22c55e">${valid.length}</div>
      <div class="stat-label">หน้าที่คำนวณได้</div>
    </div>
    <div class="stat-item">
      <div class="value" style="color:#dc2626">${noRef.length}</div>
      <div class="stat-label">ไม่มีไฟล์อ้างอิง</div>
    </div>
  </div>`;

  // Per-page cards
  results.forEach(r => {
    const hasRef = r.wer_score >= 0;
    const pct  = hasRef ? (r.wer_score * 100).toFixed(1) : null;
    const lv   = hasRef ? werLevel(r.wer_score) : null;
    const ocr  = r.ocr_text  || r.ocr_preview  || '';
    const ref  = r.reference_text || r.reference_preview || '';

    html += `
    <div class="result-card mt-3">
      <!-- Header row: page number + WER bar -->
      <div class="flex items-center gap-3 mb-3">
        <span class="font-bold text-gray-600 shrink-0">หน้าที่ ${r.page_number}</span>
        ${hasRef ? `
          <div class="flex-1 flex items-center gap-2">
            <div class="wer-bar flex-1"><div class="wer-bar-fill" style="width:${Math.min(pct,100)}%;background:${lv.color}"></div></div>
            <span class="font-bold text-sm shrink-0" style="color:${lv.color}">${pct}%</span>
            <span class="text-xs font-semibold shrink-0 px-2 py-0.5 rounded" style="background:${lv.color}18;color:${lv.color}">${lv.label}</span>
          </div>
        ` : '<span class="text-gray-400 text-xs italic">ไม่มีไฟล์อ้างอิง</span>'}
      </div>

      <!-- Image + text comparison -->
      <div class="flex gap-3 flex-wrap lg:flex-nowrap">
        ${r.image_url ? `
        <div class="shrink-0 self-start">
          <div class="text-[10px] font-bold text-gray-400 uppercase mb-1">ภาพต้นฉบับ</div>
          <img src="${r.image_url}"
               alt="Page ${r.page_number}"
               class="w-40 rounded-lg border border-gray-200 shadow-sm object-contain cursor-pointer hover:opacity-90 transition-opacity"
               onclick="window.open('${r.image_url}', '_blank')"
               onerror="this.parentElement.style.display='none'">
        </div>` : ''}
        <div class="flex-1 min-w-0">
          <div class="text-[10px] font-bold text-indigo-600 uppercase mb-1">OCR Output</div>
          <div class="text-[11px] text-gray-600 leading-relaxed whitespace-pre-wrap bg-gray-50 rounded-lg p-3 max-h-64 overflow-y-auto border border-gray-100">${escapeHtml(ocr) || '<em class="text-gray-400">ไม่มีข้อมูล</em>'}</div>
        </div>
        <div class="flex-1 min-w-0">
          <div class="text-[10px] font-bold text-green-600 uppercase mb-1">Reference (best_ocr)</div>
          <div class="text-[11px] text-gray-600 leading-relaxed whitespace-pre-wrap bg-green-50 rounded-lg p-3 max-h-64 overflow-y-auto border border-green-100">${escapeHtml(ref) || '<em class="text-gray-400">ไม่มีไฟล์อ้างอิง</em>'}</div>
        </div>
      </div>
    </div>`;
  });

  container.innerHTML = html;
}
