// Simula o modelo Gemma gerando diálogo a partir da notícia

async function gerarDialogo() {
  const falas = [
    `<strong>Sagredo:</strong> Eu li os jornais hoje, ô, caras. Parece que a China está construindo um supercomputador no espaço.`,
    `<strong>Salvati:</strong> Isso representa a mais recente etapa da automação da vigilância. O capital agora opera até fora da Terra.`,
    `<strong>Simplicio:</strong> Mas será que isso não representa progresso? Eles estão apenas inovando com liberdade.`,
    `<strong>Sagredo:</strong> Liberdade ou dominação computacional orbital? Eis a questão.`,
    `<strong>Salvati:</strong> A infraestrutura de dominação segue se estendendo. E o proletariado digital continua sem voz.`,
    `<strong>Simplicio:</strong> O importante é que mantenhamos a ordem e a soberania nacional.`,
  ];

  document.getElementById("falas").innerHTML = falas.map(f => `<p>${f}</p>`).join("");
}
