document.addEventListener("DOMContentLoaded", async () => {
  // Carrega a notícia assim que a página é aberta
  const noticia = await window.obterNoticia();
  if (noticia) {
    // Gera o primeiro diálogo automaticamente
    gerarDialogo(noticia);
  }
});

async function gerarDialogo(noticiaData) {
  const falasContainer = document.getElementById("falas");
  falasContainer.innerHTML = "<p><em>Gerando diálogo com IA... por favor, aguarde.</em></p>";

  // Se a notícia ainda não foi carregada, busca novamente
  let noticia = noticiaData;
  if (!noticia) {
    noticia = await window.obterNoticia();
    if (!noticia) {
      falasContainer.innerHTML = "<p>Não foi possível obter uma notícia para gerar o diálogo.</p>";
      return;
    }
  }

  // Definição das personalidades para enviar à IA
  const prompt = `
    Você é um roteirista que cria diálogos filosóficos.
    A notícia é:
    Título: "${noticia.titulo}"
    Conteúdo: "${noticia.conteudo}"

    Crie um diálogo curto entre três personagens sobre esta notícia. Siga estritamente as personalidades abaixo:
    - Sagredo: Observador irônico e inquieto, sensível ao absurdo moderno. Ele sempre começa o diálogo com a frase "Eu li os jornais hoje, ô, caras."
    - Salvati: Marxista pós-moderno, crítico da lógica do capital e do controle algorítmico.
    - Simplicio: Conservador confiante, defensor da ordem, da liberdade de mercado e da autoridade institucional.

    O diálogo deve ser em português do Brasil e conter pelo menos uma fala para cada personagem, mantendo o tom crítico e filosófico do projeto.
    Formato da resposta:
    <p><strong>Sagredo:</strong> [fala]</p>
    <p><strong>Salvati:</strong> [fala]</p>
    <p><strong>Simplicio:</strong> [fala]</p>
    ...
  `;

  try {
    // --- IMPORTANTE: Substitua pela chamada real à API do Gemma ---
    // O código abaixo é uma simulação de como a resposta da IA seria processada.
    // Você precisará usar o SDK ou o endpoint fetch do serviço de IA que escolher.
    const dialogoGerado = await simularChamadaGemma(prompt); // Substitua 'simularChamadaGemma' pela sua função real

    falasContainer.innerHTML = dialogoGerado;

  } catch (error) {
    console.error("Erro ao gerar diálogo com a IA:", error);
    falasContainer.innerHTML = `<p style="color: red;">Ocorreu um erro ao contatar a IA. O diálogo simulado será exibido.</p>${gerarDialogoSimulado()}`;
  }
}

/**
 * FUNÇÃO DE SIMULAÇÃO - SUBSTITUA PELA CHAMADA REAL À API
 * Esta função apenas simula uma chamada à API do Gemma.
 */
async function simularChamadaGemma(prompt) {
  console.log("Enviando o seguinte prompt para a IA (simulação):", prompt);
  // Simulação de uma resposta da IA baseada na notícia do supercomputador orbital
  const respostaSimulada = `
    <p><strong>Sagredo:</strong> Eu li os jornais hoje, ô, caras. A China agora quer processar dados no espaço, longe dos nossos olhos.</p>
    <p><strong>Salvati:</strong> É a expansão inevitável do complexo industrial-tecnológico. O capital não conhece fronteiras, nem mesmo a gravidade. A vigilância se torna celestial.</p>
    <p><strong>Simplicio:</strong> Vejo como um avanço notável. A competição impulsiona a inovação. Se não limitarmos o progresso com ideologias, todos nos beneficiaremos da soberania tecnológica.</p>
    <p><strong>Sagredo:</strong> Soberania ou o panóptico orbital definitivo? Fico imaginando o absurdo de um erro de '404 Not Found' em plena órbita.</p>
  `;
  return new Promise(resolve => setTimeout(() => resolve(respostaSimulada), 1500));
}

function gerarDialogoSimulado() {
  const falas = [
    `<strong>Sagredo:</strong> Eu li os jornais hoje, ô, caras. Parece que a China está construindo um supercomputador no espaço.`,
    `<strong>Salvati:</strong> Isso representa a mais recente etapa da automação da vigilância. O capital agora opera até fora da Terra.`,
    `<strong>Simplicio:</strong> Mas será que isso não representa progresso? Eles estão apenas inovando com liberdade.`,
    `<strong>Sagredo:</strong> Liberdade ou dominação computacional orbital? Eis a questão.`,
  ];
  return falas.map(f => `<p>${f}</p>`).join("");
}

// Associa a função ao botão, garantindo que o diálogo seja gerado sob demanda
document.addEventListener('DOMContentLoaded', () => {
    const botao = document.querySelector('button');
    if (botao) {
        botao.onclick = () => gerarDialogo(null); // Passa null para forçar a busca de uma nova notícia, se necessário
    }
});
