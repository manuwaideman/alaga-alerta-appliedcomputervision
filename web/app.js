// Alaga Alerta — front de demonstração do MVP.
// Envia a foto para a API de visão e mostra as ocorrências confirmadas no mapa.
// Para apontar para uma API em outro endereço: defina window.ALAGA_API antes deste script.
const API = (window.ALAGA_API || "").replace(/\/$/, "");

const NIVEIS = {
  alto: { cor: "#d7263d", nome: "Alto" },
  medio: { cor: "#ef8a17", nome: "Médio" },
  baixo: { cor: "#e3b505", nome: "Baixo" },
  nenhum: { cor: "#5b6b7a", nome: "Sem risco" },
};
const ROTULOS = {
  rua_alagada: "Rua alagada",
  bueiro_entupido: "Bueiro entupido",
  veiculo_submerso: "Veículo submerso",
};

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// ------------------------------------------------------------------ mapa
const mapa = L.map("mapa", { zoomControl: true }).setView([-23.5505, -46.6333], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(mapa);
const camada = L.layerGroup().addTo(mapa);
const marcadores = new Map();

let local = null;
let pino = null;

function definirLocal(lat, lon, origem) {
  local = { lat, lon };
  if (!pino) {
    pino = L.marker([lat, lon], { draggable: true, title: "Local da foto" }).addTo(mapa);
    pino.on("dragend", () => {
      const p = pino.getLatLng();
      definirLocal(p.lat, p.lng, "mapa");
    });
  } else {
    pino.setLatLng([lat, lon]);
  }
  $("local-texto").textContent = `${origem === "gps" ? "Sua localização" : "Local marcado"}: ${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}

mapa.on("click", (e) => definirLocal(e.latlng.lat, e.latlng.lng, "mapa"));

$("btn-geo").addEventListener("click", () => {
  if (!navigator.geolocation) return toast("Geolocalização indisponível neste navegador.");
  $("btn-geo").disabled = true;
  navigator.geolocation.getCurrentPosition(
    (p) => {
      $("btn-geo").disabled = false;
      definirLocal(p.coords.latitude, p.coords.longitude, "gps");
      mapa.flyTo([p.coords.latitude, p.coords.longitude], 16);
    },
    () => {
      $("btn-geo").disabled = false;
      toast("Não foi possível obter sua localização. Clique no mapa para marcar o local.");
    },
    { enableHighAccuracy: true, timeout: 10000 }
  );
});

// ------------------------------------------------------------------ envio
$("foto").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const img = $("preview");
  img.src = URL.createObjectURL(f);
  img.hidden = false;
  $("upload-texto").hidden = true;
});

$("form-envio").addEventListener("submit", async (e) => {
  e.preventDefault();
  const arquivo = $("foto").files[0];
  if (!arquivo) return toast("Escolha uma foto primeiro.");

  const dados = new FormData();
  dados.append("imagem", arquivo);
  dados.append("origem", "cidadao");
  if (local) {
    dados.append("latitude", local.lat);
    dados.append("longitude", local.lon);
  }
  const desc = $("descricao").value.trim();
  if (desc) dados.append("descricao", desc);

  const btn = $("btn-enviar");
  btn.disabled = true;
  btn.textContent = "Analisando foto…";
  try {
    const resp = await fetch(`${API}/api/deteccoes`, { method: "POST", body: dados });
    const corpo = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const det = Array.isArray(corpo.detail) ? corpo.detail.map((d) => d.msg).join("; ") : corpo.detail;
      throw new Error(det || `Erro ${resp.status}`);
    }
    mostrarResultado(corpo);
    if (corpo.confirmada) {
      await carregarOcorrencias();
      const id = corpo.ocorrencia.id;
      const m = marcadores.get(id);
      if (m) {
        mapa.flyTo(m.getLatLng(), 16);
        m.openPopup();
      }
    }
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Enviar foto";
  }
});

function mostrarResultado(r) {
  $("resultado").hidden = false;
  $("res-titulo").textContent = r.confirmada ? "Ocorrência confirmada" : "Nada confirmado";
  const risco = $("res-risco");
  const nivel = r.confirmada ? r.risco.nivel : "nenhum";
  risco.className = `risco ${nivel}`;
  risco.textContent = r.confirmada ? `Risco ${NIVEIS[nivel].nome} · ${r.risco.score}` : "Sem ocorrência";
  $("res-msg").textContent = r.confirmada ? r.risco.motivo : r.mensagem;

  const img = $("res-imagem");
  if (r.ocorrencia) {
    img.src = r.ocorrencia.imagem_anotada_url;
    img.hidden = false;
  } else {
    img.hidden = true;
  }

  const lim = r.ocorrencia ? Math.min(...r.ocorrencia.deteccoes.map((d) => d.confianca)) : Infinity;
  $("res-deteccoes").innerHTML = r.deteccoes.length
    ? r.deteccoes
        .map(
          (d) => `<li class="${d.confianca < lim ? "abaixo" : ""}">
            <span>${esc(d.rotulo)}</span><span>${Math.round(d.confianca * 100)}%</span>
            <span class="barra"><i style="width:${Math.round(d.confianca * 100)}%"></i></span></li>`
        )
        .join("")
    : "";

  const a = r.anonimizacao;
  $("res-anon").textContent =
    `Privacidade: ${a.rostos} rosto(s) e ${a.placas} placa(s) borrados. ` +
    `Local: ${r.localizacao.fonte === "exif" ? "GPS da foto" : "informado"}. ` +
    `Tempo: ${r.tempo_ms.anonimizacao + r.tempo_ms.deteccao} ms.`;

  // Integração com a API central do MVP (só aparece quando MVP_URL/MVP_API_KEY estão configuradas)
  const mvp = $("res-mvp");
  mvp.hidden = !r.mvp;
  if (r.mvp) {
    mvp.className = `mvp ${r.mvp.publicada ? "ok" : "falha"}`;
    mvp.textContent = r.mvp.publicada
      ? `Enviada ao Alaga Alerta (MVP): bairro ${r.mvp.bairro || r.mvp.bairro_id}`
      : `Não enviada ao MVP: ${r.mvp.motivo}`;
  }
}

// ------------------------------------------------------------------ ocorrências
function tempoRelativo(iso) {
  const seg = (new Date(iso) - Date.now()) / 1000;
  const rtf = new Intl.RelativeTimeFormat("pt-BR", { numeric: "auto" });
  const abs = Math.abs(seg);
  if (abs < 60) return rtf.format(Math.round(seg), "second");
  if (abs < 3600) return rtf.format(Math.round(seg / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(seg / 3600), "hour");
  return rtf.format(Math.round(seg / 86400), "day");
}

function popupHtml(p) {
  const classes = p.classes.map((c) => ROTULOS[c] || c).join(", ");
  return `<div class="popup">
    <img src="${esc(p.imagem_anotada_url)}" alt="Foto da ocorrência">
    <p><strong>${esc(classes)}</strong></p>
    <p>Risco ${esc(NIVEIS[p.risco_nivel]?.nome || p.risco_nivel)} (${p.risco_score}) · ${esc(tempoRelativo(p.criada_em))}</p>
    <p>${esc(p.risco_motivo)}</p>
    ${p.descricao ? `<p><em>“${esc(p.descricao)}”</em></p>` : ""}
    <p class="dica">${p.origem === "camera" ? `Câmera ${esc(p.camera_id || "")}` : "Enviado por cidadão"}</p>
  </div>`;
}

async function carregarOcorrencias() {
  try {
    const resp = await fetch(`${API}/api/ocorrencias?horas=${$("janela").value}`);
    const geo = await resp.json();
    camada.clearLayers();
    marcadores.clear();
    const lista = $("lista");
    lista.innerHTML = "";
    if (!geo.features.length) {
      lista.innerHTML = '<li class="vazio">Nenhuma ocorrência nesta janela de tempo.</li>';
      return;
    }
    for (const f of geo.features) {
      const p = f.properties;
      const [lon, lat] = f.geometry.coordinates;
      const cor = NIVEIS[p.risco_nivel]?.cor || "#888";
      const m = L.circleMarker([lat, lon], {
        radius: 7 + p.risco_score / 10,
        color: "#fff",
        weight: 2,
        fillColor: cor,
        fillOpacity: 0.9,
      })
        .bindPopup(popupHtml(p), { maxWidth: 260 })
        .addTo(camada);
      marcadores.set(f.id, m);

      const li = document.createElement("li");
      li.innerHTML = `<img src="${esc(p.imagem_anotada_url)}" alt="">
        <div class="info"><strong>${esc(p.classes.map((c) => ROTULOS[c] || c).join(", "))}</strong>
        <span class="dica">Risco ${esc(NIVEIS[p.risco_nivel]?.nome)} · ${esc(tempoRelativo(p.criada_em))}</span></div>`;
      li.addEventListener("click", () => {
        mapa.flyTo([lat, lon], 16);
        m.openPopup();
      });
      lista.appendChild(li);
    }
  } catch {
    toast("Não foi possível carregar as ocorrências.");
  }
}

async function verificarModelo() {
  const chip = $("status-modelo");
  try {
    const h = await (await fetch(`${API}/api/health`)).json();
    if (h.status === "ok") {
      chip.textContent = "modelo pronto";
      chip.className = "chip ok";
      chip.title = `${h.modelo} · rostos: ${h.anonimizacao.rostos} · placas: ${h.anonimizacao.placas}`;
    } else if (h.status === "sem_anonimizacao") {
      chip.textContent = "anonimização indisponível";
      chip.className = "chip erro";
      chip.title = h.anonimizacao.erro || "";
    } else {
      chip.textContent = "modelo não carregado";
      chip.className = "chip erro";
      chip.title = h.erro_modelo || "";
    }
  } catch {
    chip.textContent = "API offline";
    chip.className = "chip erro";
  }
}

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 5000);
}

$("janela").addEventListener("change", carregarOcorrencias);
verificarModelo();
carregarOcorrencias();
setInterval(carregarOcorrencias, 30000);
