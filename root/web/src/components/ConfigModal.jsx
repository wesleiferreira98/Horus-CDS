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
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-2xl w-full max-w-sm overflow-hidden">
        {/* Header */}
        <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-800">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">Configurações</h2>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Tipo de gráfico — Logs de Rede
            </label>
            <select
              value={localType}
              onChange={(e) => setLocalType(e.target.value)}
              className="w-full bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700
                         rounded-xl px-3 py-2.5 text-sm text-gray-900 dark:text-gray-100
                         focus:outline-none focus:ring-2 focus:ring-cyan-500 transition"
            >
              <option value="bar">Barras</option>
              <option value="pie">Pizza</option>
            </select>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-gray-100 dark:border-gray-800 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-sm font-semibold
                       bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300
                       hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            Cancelar
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-xl text-sm font-semibold
                       bg-cyan-600 hover:bg-cyan-700 text-white transition-colors"
          >
            Salvar
          </button>
        </div>
      </div>
    </div>
  )
}
