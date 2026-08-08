import { useState } from 'react'

export default function ConfigModal({ open, onClose, chartType, onSave }) {
  const [localType, setLocalType] = useState(chartType)

  if (!open) return null

  const handleSave = () => {
    onSave(localType)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-white dark:bg-gray-900 rounded-2xl border border-brand-muted/40 dark:border-gray-700 shadow-2xl w-full max-w-sm overflow-hidden">
        {/* Header */}
        <div className="px-6 py-5 border-b border-brand-muted/20 dark:border-gray-800">
          <h2 className="text-lg font-bold text-brand-primary dark:text-white">Configurações</h2>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-semibold text-brand-text dark:text-gray-300 mb-2">
              Tipo de gráfico — Logs de Rede
            </label>
            <select
              value={localType}
              onChange={(e) => setLocalType(e.target.value)}
              className="w-full bg-brand-bg dark:bg-gray-800 border border-brand-muted/30 dark:border-gray-700
                         rounded-xl px-3 py-2.5 text-sm text-brand-text dark:text-gray-100
                         focus:outline-none focus:ring-2 focus:ring-brand-primary transition"
            >
              <option value="bar">Barras</option>
              <option value="pie">Pizza</option>
            </select>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-brand-muted/20 dark:border-gray-800 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-sm font-semibold
                       bg-brand-bg dark:bg-gray-800 text-brand-text dark:text-gray-300 border border-brand-muted/30
                       hover:bg-brand-primary/10 hover:text-brand-primary dark:hover:bg-gray-700 transition-colors"
          >
            Cancelar
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-xl text-sm font-semibold
                       bg-brand-primary hover:bg-brand-hover text-white transition-colors"
          >
            Salvar
          </button>
        </div>
      </div>
    </div>
  )
}
