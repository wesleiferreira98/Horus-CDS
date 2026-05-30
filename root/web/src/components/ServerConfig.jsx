function PlayIcon() {
  return (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <polygon points="5,3 19,12 5,21" />
    </svg>
  )
}

function PauseIcon() {
  return (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <rect x="6" y="4" width="4" height="16" />
      <rect x="14" y="4" width="4" height="16" />
    </svg>
  )
}

function SyncIcon({ spinning }) {
  return (
    <svg className={`w-4 h-4 ${spinning ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

export default function ServerConfig({
  apiBase,
  monitoring, onToggleMonitor, onSync,
  apiStatus, onToggleMode,
  loading, connected, lastUpdated,
}) {
  const simulationMode = apiStatus?.simulation_mode ?? null

  const formattedTime = lastUpdated
    ? lastUpdated.toLocaleTimeString('pt-BR')
    : null

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-5 space-y-4">
      {/* URL da API + status + timestamp */}
      <div className="flex gap-3 items-center flex-wrap">
        <div className="flex-1 flex items-center gap-2.5 px-4 py-2.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl min-w-0">
          <span className="text-xs font-semibold text-gray-400 dark:text-gray-500 shrink-0 uppercase tracking-wide">
            API
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-300 font-mono truncate">
            {apiBase ?? '…'}
          </span>
        </div>

        <div className={`flex items-center gap-2 text-xs font-semibold px-3 py-2.5 rounded-xl border shrink-0 transition-colors ${
          connected
            ? 'bg-emerald-900/20 text-emerald-400 border-emerald-700/50'
            : 'bg-gray-100 dark:bg-gray-800 text-gray-400 border-gray-200 dark:border-gray-700'
        }`}>
          <span className={`w-2 h-2 rounded-full shrink-0 ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-gray-400'}`} />
          {connected ? 'Online' : 'Offline'}
        </div>

        {formattedTime && (
          <div className="flex items-center gap-1.5 text-xs text-gray-400 dark:text-gray-500 shrink-0">
            <ClockIcon />
            <span>Atualizado às {formattedTime}</span>
          </div>
        )}
      </div>

      {/* Controles */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-2">
          <button
            onClick={onToggleMonitor}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
              monitoring
                ? 'bg-red-600 hover:bg-red-700 text-white shadow-lg shadow-red-900/30'
                : 'bg-cyan-600 hover:bg-cyan-700 text-white shadow-lg shadow-cyan-900/30'
            }`}
          >
            {monitoring ? <PauseIcon /> : <PlayIcon />}
            {monitoring ? 'Parar' : 'Iniciar'}
          </button>

          <button
            onClick={onSync}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold
                       bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300
                       hover:bg-gray-200 dark:hover:bg-gray-700
                       disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <SyncIcon spinning={loading} />
            Atualizar
          </button>
        </div>

        {apiStatus && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-500 dark:text-gray-400 font-medium">Modo:</span>
            <div className="flex rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
              <button
                onClick={() => onToggleMode(true)}
                className={`px-3 py-1.5 font-semibold transition-colors ${
                  simulationMode === true
                    ? 'bg-amber-500 text-white'
                    : 'bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                Simulação
              </button>
              <button
                onClick={() => onToggleMode(false)}
                className={`px-3 py-1.5 font-semibold transition-colors border-l border-gray-200 dark:border-gray-700 ${
                  simulationMode === false
                    ? 'bg-cyan-600 text-white'
                    : 'bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                Captura Real
              </button>
            </div>
          </div>
        )}
      </div>

      {monitoring && (
        <p className="text-xs text-cyan-500 dark:text-cyan-400">
          Atualizando automaticamente a cada 10 segundos...
        </p>
      )}
    </div>
  )
}
