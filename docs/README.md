# Documentação - Horus-CDS

Esta pasta contém toda a documentação adicional do projeto Horus-CDS.

## Estrutura da Documentação

### Guias de Instalação

* **[QUICKSTART.md](QUICKSTART.md)** - Guia rápido de instalação e primeiros passos
  - Instalação automática com scripts
  - Instalação manual (Docker, venv)
  - Comandos úteis
  - Troubleshooting rápido

* **[CUDA_INSTALLATION.md](CUDA_INSTALLATION.md)** - Guia completo de instalação CUDA e cuDNN
  - Instruções para Linux (Ubuntu/Fedora)
  - Instruções para Windows 10/11
  - Instruções para macOS
  - NVIDIA Container Toolkit para Docker
  - Verificação e troubleshooting
  - Tabelas de compatibilidade

### Guias de Deployment

* **[DOCKER.md](DOCKER.md)** - Guia completo de deployment com Docker
  - Instalação do Docker
  - Suporte GPU com NVIDIA Container Toolkit
  - Comandos de gerenciamento
  - Monitoramento e logs
  - Backup e restore
  - Troubleshooting Docker

### Documentação de Scripts

* **[INSTALL_SCRIPTS.md](INSTALL_SCRIPTS.md)** - Documentação dos scripts de instalação
  - Detalhes técnicos dos scripts
  - Funcionalidades e fluxo
  - Exemplos de uso
  - Resolução de problemas

### Registros e Sumários

* **[CHANGELOG_CUDA.md](CHANGELOG_CUDA.md)** - Registro de mudanças da documentação CUDA
  - Lista de arquivos criados/modificados
  - Benefícios implementados
  - Compatibilidade testada
  - Próximos passos

* **[SUMMARY_CUDA_DOCS.md](SUMMARY_CUDA_DOCS.md)** - Sumário executivo da documentação
  - Visão geral completa
  - Métricas e estatísticas
  - Casos de uso
  - Links de referência

## Como Usar Esta Documentação

### Para Iniciantes

1. Comece com **[QUICKSTART.md](QUICKSTART.md)** para instalação rápida
2. Use os scripts de instalação em `../scripts/`
3. Consulte **[DOCKER.md](DOCKER.md)** se preferir containers

### Para Usuários com GPU

1. Leia **[CUDA_INSTALLATION.md](CUDA_INSTALLATION.md)** primeiro
2. Instale drivers NVIDIA, CUDA e cuDNN
3. Use **[DOCKER.md](DOCKER.md)** para deployment com GPU

### Para Deployment em Produção

1. Consulte **[DOCKER.md](DOCKER.md)** para configuração
2. Revise **[INSTALL_SCRIPTS.md](INSTALL_SCRIPTS.md)** para automação
3. Verifique **[CUDA_INSTALLATION.md](CUDA_INSTALLATION.md)** para otimizações GPU

## Navegação

- **Voltar para documentação principal**: [../README.md](../README.md)
- **Ver scripts de instalação**: [../scripts/](../scripts/)
- **Documentação da API**: [../root/API/README_API.md](../root/API/README_API.md)

## Contribuindo com a Documentação

Para adicionar ou atualizar documentação:

1. Mantenha o formato Markdown consistente
2. Use linguagem formal sem emojis
3. Inclua exemplos práticos
4. Atualize este README se criar novos documentos
5. Adicione referências cruzadas quando apropriado

## Suporte

Para dúvidas sobre a documentação:
- Abra uma issue no GitHub
- Consulte a documentação principal
- Verifique os exemplos nos guias

---

**Última atualização**: Novembro 2025
