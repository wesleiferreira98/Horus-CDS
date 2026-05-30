function MetricItem({ label, value, desc, accent }) {
  const accents = {
    blue:   'text-blue-500 dark:text-blue-400',
    red:    'text-red-500 dark:text-red-400',
    cyan:   'text-cyan-500 dark:text-cyan-400',
    purple: 'text-purple-500 dark:text-purple-400',
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 px-4 py-3 flex items-center gap-4">
      <div className="min-w-0">
        <p className="text-xs text-gray-500 dark:text-gray-400 font-medium truncate">{label}</p>
        <p className={`text-xl font-bold mt-0.5 truncate ${accents[accent]}`}>{value}</p>
        <p className="text-xs text-gray-400 dark:text-gray-600 mt-0.5 truncate">{desc}</p>
      </div>
    </div>
  )
}

export default function MetricsSummary({ data, apiStatus }) {
  const ataques     = data?.packet_logs?.ataques_detectados  ?? 0
  const permitidas  = data?.packet_logs?.requisicoes_permitidas ?? 0
  const inconclusivos = data?.packet_logs?.inconclusivos     ?? 0
  const total       = ataques + permitidas + inconclusivos

  const taxaAtaque  = total > 0 ? ((ataques / total) * 100).toFixed(1) : '—'
  const totalPred   = data?.predictions_log?.resultados?.length ?? 0

  const modelRaw    = apiStatus?.active_model ?? ''
  const modelMatch  = modelRaw.match(/V(\d+)\s*\((\w+)\)/)
  const modelShort  = modelMatch ? `V${modelMatch[1]} · ${modelMatch[2]}` : (modelRaw || '—')

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <MetricItem
        label="Total Analisados"
        value={total > 0 ? total.toLocaleString('pt-BR') : '—'}
        desc="pacotes acumulados"
        accent="blue"
      />
      <MetricItem
        label="Taxa de Ataques"
        value={taxaAtaque !== '—' ? `${taxaAtaque}%` : '—'}
        desc="do tráfego capturado"
        accent="red"
      />
      <MetricItem
        label="Predições no Buffer"
        value={totalPred > 0 ? totalPred.toLocaleString('pt-BR') : '—'}
        desc="resultados registrados"
        accent="cyan"
      />
      <MetricItem
        label="Modelo Ativo"
        value={modelShort || '—'}
        desc="limiar de ataque: 200"
        accent="purple"
      />
    </div>
  )
}
