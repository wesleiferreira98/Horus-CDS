OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Hórus-CDS API",
        "description": (
            "API de detecção de intrusão em redes de smart grid com modelos de deep learning. "
            "Suporta captura real de pacotes (via Scapy/root) e modo de simulação de tráfego."
        ),
        "version": "1.0.0",
        "contact": {
            "name": "Hórus-CDS",
        },
    },
    "servers": [
        {"url": "http://localhost:5000", "description": "Servidor local"}
    ],
    "tags": [
        {"name": "Monitoramento", "description": "Leitura de dados e logs de captura"},
        {"name": "Modelos",       "description": "Gerenciamento do modelo de ML ativo"},
        {"name": "Simulação",     "description": "Controle do modo e configuração de simulação"},
        {"name": "Predição",      "description": "Inferência direta via features brutas"},
        {"name": "Gráficos",      "description": "Imagens PNG dos gráficos gerados pelo servidor"},
    ],
    "paths": {

        # ─── GET /status ────────────────────────────────────────────────────────
        "/status": {
            "get": {
                "tags": ["Monitoramento"],
                "summary": "Status da API",
                "description": (
                    "Retorna informações sobre o modelo ativo, modo de captura "
                    "(simulação vs. real) e permissões do processo."
                ),
                "operationId": "getStatus",
                "responses": {
                    "200": {
                        "description": "Status retornado com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Status"},
                                "example": {
                                    "active_model": "Horus-CDS V4 (TCN)",
                                    "simulation_mode": True,
                                    "scapy_available": False,
                                    "running_as_root": False,
                                    "mode_description": "Simulação (sem sudo)",
                                },
                            }
                        },
                    }
                },
            }
        },

        # ─── GET /gerar_dados ───────────────────────────────────────────────────
        "/gerar_dados": {
            "get": {
                "tags": ["Monitoramento"],
                "summary": "Dados de logs e predições",
                "description": (
                    "Retorna os logs de pacotes capturados e as predições do modelo ativo. "
                    "Use o parâmetro `filter` para ordenar ou limitar os registros."
                ),
                "operationId": "gerarDados",
                "parameters": [
                    {
                        "name": "filter",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["recentes", "antigos", "todos"],
                            "default": "todos",
                        },
                        "description": "Ordenação dos logs retornados.",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Dados retornados com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DadosResponse"},
                                "example": {
                                    "packet_logs": {
                                        "ataques_detectados": 5,
                                        "requisicoes_permitidas": 42,
                                        "inconclusivos": 2,
                                        "logs": [
                                            {
                                                "timestamp": "2024-01-01 12:00:00",
                                                "source_ip": "192.168.1.10",
                                                "destination_ip": "192.168.1.1",
                                                "tipo": "Ataque",
                                            }
                                        ],
                                    },
                                    "predictions_log": {
                                        "resultados": [1, 0, 1],
                                        "normalized_predictions": [0.91, 0.12, 0.88],
                                        "desnormalized_predictions": [320.5, 80.3, 295.1],
                                    },
                                },
                            }
                        },
                    }
                },
            }
        },

        # ─── POST /predict ──────────────────────────────────────────────────────
        "/predict": {
            "post": {
                "tags": ["Predição"],
                "summary": "Inferência direta",
                "description": (
                    "Executa o modelo ativo sobre um vetor de features e retorna "
                    "a predição normalizada e desnormalizada."
                ),
                "operationId": "predict",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PredictRequest"},
                            "example": {"features": [0.1, 0.5, 0.3, 0.9, 0.2, 0.7]},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Predição realizada com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PredictResponse"},
                                "example": {
                                    "prediction": 1,
                                    "normalized": 0.87,
                                    "desnormalized": 310.4,
                                },
                            }
                        },
                    },
                    "400": {
                        "description": "Features inválidas ou ausentes",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                                "example": {"error": "Features inválidas."},
                            }
                        },
                    },
                    "500": {
                        "description": "Erro interno na inferência",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                            }
                        },
                    },
                },
            }
        },

        # ─── POST /set_model ────────────────────────────────────────────────────
        "/set_model": {
            "post": {
                "tags": ["Modelos"],
                "summary": "Altera o modelo ativo",
                "description": (
                    "Troca o modelo de deep learning usado nas predições. "
                    "O modelo anterior é descarregado da memória antes de carregar o novo."
                ),
                "operationId": "setModel",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SetModelRequest"},
                            "example": {"model": "Horus-CDS V4 (TCN)"},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Modelo alterado com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MessageResponse"},
                                "example": {"message": "Modelo alterado para Horus-CDS V4 (TCN) com sucesso."},
                            }
                        },
                    },
                    "400": {
                        "description": "Modelo não especificado",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                                "example": {"error": "Nenhum modelo foi especificado."},
                            }
                        },
                    },
                    "404": {
                        "description": "Arquivo do modelo não encontrado em disco",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                            }
                        },
                    },
                    "500": {
                        "description": "Erro ao carregar o modelo",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                            }
                        },
                    },
                },
            }
        },

        # ─── POST /set_simulation_config ────────────────────────────────────────
        "/set_simulation_config": {
            "post": {
                "tags": ["Simulação"],
                "summary": "Configura o tráfego simulado",
                "description": (
                    "Define o perfil de tráfego gerado no modo simulação: "
                    "100% ataques, 100% permitido, ou misto com proporção configurável."
                ),
                "operationId": "setSimulationConfig",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SimConfigRequest"},
                            "example": {"mode": "mixed", "attack_ratio": 0.4},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Configuração aplicada com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MessageResponse"},
                                "example": {"message": "Simulação configurada: Misto (40% ataques)"},
                            }
                        },
                    },
                    "400": {
                        "description": "Modo inválido",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                                "example": {"error": "Modo inválido. Use: attack, allowed ou mixed"},
                            }
                        },
                    },
                },
            }
        },

        # ─── POST /toggle_mode ──────────────────────────────────────────────────
        "/toggle_mode": {
            "post": {
                "tags": ["Simulação"],
                "summary": "Alterna entre simulação e captura real",
                "description": (
                    "Troca o modo de operação da captura de pacotes. "
                    "Captura real exige que o processo rode com privilégios de root (sudo). "
                    "Retorna 403 se tentar ativar captura real sem as permissões necessárias."
                ),
                "operationId": "toggleMode",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ToggleModeRequest"},
                            "example": {"simulation_mode": False},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Modo alterado com sucesso",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ToggleModeResponse"},
                                "example": {
                                    "message": "Modo alterado para: Captura Real (com sudo)",
                                    "simulation_mode": False,
                                },
                            }
                        },
                    },
                    "403": {
                        "description": "Sem permissão de root para ativar captura real",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                                "example": {
                                    "error": "Modo real requer permissões de root.",
                                    "current_mode": "simulation",
                                },
                            }
                        },
                    },
                    "500": {
                        "description": "Erro ao reiniciar a captura",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"},
                            }
                        },
                    },
                },
            }
        },

        # ─── GET /prediction_chart ──────────────────────────────────────────────
        "/prediction_chart": {
            "get": {
                "tags": ["Gráficos"],
                "summary": "Gráfico de predições",
                "description": "Retorna uma imagem PNG com o histórico de predições do modelo ativo.",
                "operationId": "predictionChart",
                "responses": {
                    "200": {
                        "description": "Imagem PNG do gráfico",
                        "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
                    }
                },
            }
        },

        # ─── GET /log_chart ─────────────────────────────────────────────────────
        "/log_chart": {
            "get": {
                "tags": ["Gráficos"],
                "summary": "Gráfico de logs de rede",
                "description": "Retorna uma imagem PNG com a distribuição dos pacotes capturados.",
                "operationId": "logChart",
                "responses": {
                    "200": {
                        "description": "Imagem PNG do gráfico",
                        "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
                    }
                },
            }
        },
    },

    # ─── Schemas reutilizáveis ───────────────────────────────────────────────
    "components": {
        "schemas": {

            "Status": {
                "type": "object",
                "properties": {
                    "active_model":      {"type": "string", "example": "Horus-CDS V4 (TCN)"},
                    "simulation_mode":   {"type": "boolean", "example": True},
                    "scapy_available":   {"type": "boolean", "example": False},
                    "running_as_root":   {"type": "boolean", "example": False},
                    "mode_description":  {"type": "string",  "example": "Simulação (sem sudo)"},
                },
            },

            "PacketLog": {
                "type": "object",
                "properties": {
                    "timestamp":      {"type": "string", "example": "2024-01-01 12:00:00"},
                    "source_ip":      {"type": "string", "example": "192.168.1.10"},
                    "destination_ip": {"type": "string", "example": "192.168.1.1"},
                    "tipo":           {"type": "string", "enum": ["Ataque", "Permitido", "Inconclusivo"]},
                },
            },

            "PacketLogs": {
                "type": "object",
                "properties": {
                    "ataques_detectados":    {"type": "integer", "example": 5},
                    "requisicoes_permitidas":{"type": "integer", "example": 42},
                    "inconclusivos":         {"type": "integer", "example": 2},
                    "logs": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/PacketLog"},
                    },
                },
            },

            "PredictionsLog": {
                "type": "object",
                "properties": {
                    "resultados": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Classificação binária: 1 = ataque, 0 = normal",
                        "example": [1, 0, 1],
                    },
                    "normalized_predictions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Saída normalizada do modelo (0.0 – 1.0)",
                        "example": [0.91, 0.12, 0.88],
                    },
                    "desnormalized_predictions": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Saída desnormalizada em escala original",
                        "example": [320.5, 80.3, 295.1],
                    },
                },
            },

            "DadosResponse": {
                "type": "object",
                "properties": {
                    "packet_logs":     {"$ref": "#/components/schemas/PacketLogs"},
                    "predictions_log": {"$ref": "#/components/schemas/PredictionsLog"},
                },
            },

            "PredictRequest": {
                "type": "object",
                "required": ["features"],
                "properties": {
                    "features": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Vetor de features numéricas para o modelo",
                        "example": [0.1, 0.5, 0.3, 0.9, 0.2, 0.7],
                    }
                },
            },

            "PredictResponse": {
                "type": "object",
                "properties": {
                    "prediction":     {"type": "integer",  "example": 1},
                    "normalized":     {"type": "number",   "example": 0.87},
                    "desnormalized":  {"type": "number",   "example": 310.4},
                },
            },

            "SetModelRequest": {
                "type": "object",
                "required": ["model"],
                "properties": {
                    "model": {
                        "type": "string",
                        "enum": [
                            "Horus-CDS V1 (RNN)",
                            "Horus-CDS V2 (LSTM)",
                            "Horus-CDS V3 (GRU)",
                            "Horus-CDS V4 (TCN)",
                        ],
                        "example": "Horus-CDS V4 (TCN)",
                    }
                },
            },

            "SimConfigRequest": {
                "type": "object",
                "required": ["mode"],
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["attack", "allowed", "mixed"],
                        "description": "Perfil de tráfego: attack=100% ataques, allowed=100% normal, mixed=proporcional",
                        "example": "mixed",
                    },
                    "attack_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "Proporção de pacotes de ataque no modo mixed (0.0 – 1.0)",
                        "example": 0.4,
                    },
                },
            },

            "ToggleModeRequest": {
                "type": "object",
                "required": ["simulation_mode"],
                "properties": {
                    "simulation_mode": {
                        "type": "boolean",
                        "description": "true = simulação, false = captura real (requer root)",
                        "example": False,
                    }
                },
            },

            "ToggleModeResponse": {
                "type": "object",
                "properties": {
                    "message":         {"type": "string", "example": "Modo alterado para: Captura Real (com sudo)"},
                    "simulation_mode": {"type": "boolean", "example": False},
                },
            },

            "MessageResponse": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
            },

            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"}
                },
            },
        }
    },
}
