// Simula uma API que pega uma notícia relevante da área de ciência/tecnologia/economia

async function obterNoticia() {
  const noticiaFalsa = {
    titulo: "China anuncia supercomputador orbital para IA espacial",
    conteudo:
      "A China anunciou nesta semana o início da construção de um supercomputador no espaço, projetado para executar modelos de inteligência artificial avançados em órbita. O objetivo é superar as limitações energéticas e térmicas dos sistemas terrestres.",
  };

  const container = document.getElementById("noticia-texto");
  container.innerHTML = `<h3>${noticiaFalsa.titulo}</h3><p>${noticiaFalsa.conteudo}</p>`;

  return noticiaFalsa;
}

window.onload = obterNoticia;
