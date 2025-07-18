// Busca a notícia mais recente do feed RSS de Tecnologia do G1 usando um conversor RSS-to-JSON

async function obterNoticia() {
  const container = document.getElementById("noticia-texto");
  container.innerHTML = "<h4>Buscando notícia relevante...</h4>";

  try {
    // URL do feed RSS do G1 Tecnologia, codificada para uso na API do rss2json
    const rssUrl = "https://g1.globo.com/rss/g1/tecnologia/";
    const apiUrl = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssUrl)}`;

    const response = await fetch(apiUrl);
    const data = await response.json();

    if (data.status === "ok" && data.items.length > 0) {
      const noticiaRecente = data.items[0]; // Pega a notícia mais recente do feed

      const noticia = {
        titulo: noticiaRecente.title,
        conteudo: noticiaRecente.description,
        link: noticiaRecente.link,
      };

      container.innerHTML = `<h3>${noticia.titulo}</h3><p>${noticia.conteudo}</p><p><a href="${noticia.link}" target="_blank">Leia mais</a></p>`;

      // Retorna o título e o conteúdo para serem usados na geração do diálogo
      return {
        titulo: noticia.titulo,
        conteudo: noticia.conteudo
      };
    } else {
      throw new Error("Não foi possível carregar as notícias.");
    }
  } catch (error) {
    console.error("Erro ao buscar notícia:", error);
    container.innerHTML = `<p style="color: red;">Falha ao carregar a notícia. Tente novamente mais tarde.</p>`;
    return null;
  }
}

// Expõe a função para ser chamada pelo script.js
window.obterNoticia = obterNoticia;
