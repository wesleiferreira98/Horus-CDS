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
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-white dark:bg-gray-900 rounded-2xl border border-brand-muted/40 dark:border-gray-700 shadow-2xl w-full max-w-md overflow-hidden">
        {/* Header */}
        <div className="px-6 py-5 border-b border-brand-muted/20 dark:border-gray-800">
          <h2 className="text-lg font-bold text-brand-primary dark:text-white">Selecionar Modelo</h2>
          <p className="text-sm text-brand-soft dark:text-gray-400 mt-1">
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
                    ? 'bg-brand-bg dark:bg-brand-primary/20 border-brand-primary/40 dark:border-brand-primary/50'
                    : 'bg-brand-bg/50 dark:bg-gray-800 border-brand-muted/20 dark:border-gray-700 hover:bg-brand-bg dark:hover:bg-gray-700'
                }`}
              >
                <div>
                  <div className={`font-semibold text-sm ${isActive ? 'text-brand-primary dark:text-white' : 'text-brand-text dark:text-gray-200'}`}>
                    {model.name}
                  </div>
                  <div className="text-xs text-brand-soft dark:text-gray-400 mt-0.5">{model.desc}</div>
                </div>
                {isActive && (
                  <span className="shrink-0 ml-3 text-xs bg-brand-primary/10 dark:bg-brand-primary/30
                                   text-brand-primary dark:text-white border border-brand-primary/30 dark:border-brand-primary/50
                                   px-2.5 py-0.5 rounded-full font-semibold">
                    Ativo
                  </span>
                )}
              </button>
            )
          })}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-brand-muted/20 dark:border-gray-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-sm font-semibold
                       bg-brand-bg dark:bg-gray-800 text-brand-text dark:text-gray-300 border border-brand-muted/30
                       hover:bg-brand-primary/10 hover:text-brand-primary dark:hover:bg-gray-700 transition-colors"
          >
            Fechar
          </button>
        </div>
      </div>
    </div>
  )
}
