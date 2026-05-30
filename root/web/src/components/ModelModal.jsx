const MODELS = [
  { name: 'Horus-CDS V1 (RNN)', type: 'RNN', desc: 'Rede Neural Recorrente' },
  { name: 'Horus-CDS V2 (LSTM)', type: 'LSTM', desc: 'Long Short-Term Memory' },
  { name: 'Horus-CDS V3 (GRU)', type: 'GRU', desc: 'Gated Recurrent Unit' },
  { name: 'Horus-CDS V4 (TCN)', type: 'TCN', desc: 'Temporal Convolutional Network' },
]

export default function ModelModal({ open, onClose, currentModel, onSelect }) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-2xl w-full max-w-md overflow-hidden">
        {/* Header */}
        <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-800">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">Selecionar Modelo</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Escolha o modelo de detecção ativo
          </p>
        </div>

        {/* Model list */}
        <div className="p-5 space-y-2">
          {MODELS.map((model) => {
            const isActive = currentModel === model.name
            return (
              <button
                key={model.name}
                onClick={() => onSelect(model.name)}
                className={`w-full flex items-center justify-between p-4 rounded-xl border text-left transition-all ${
                  isActive
                    ? 'bg-cyan-50 dark:bg-cyan-900/20 border-cyan-300 dark:border-cyan-700/60'
                    : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <div>
                  <div className={`font-semibold text-sm ${isActive ? 'text-cyan-700 dark:text-cyan-300' : 'text-gray-800 dark:text-gray-200'}`}>
                    {model.name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{model.desc}</div>
                </div>
                {isActive && (
                  <span className="shrink-0 ml-3 text-xs bg-cyan-100 dark:bg-cyan-900/50 text-cyan-700 dark:text-cyan-400 border border-cyan-300 dark:border-cyan-700/50 px-2.5 py-0.5 rounded-full font-semibold">
                    Ativo
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-gray-100 dark:border-gray-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-sm font-semibold
                       bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300
                       hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            Fechar
          </button>
        </div>
      </div>
    </div>
  )
}
