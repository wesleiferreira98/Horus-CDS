import { useState } from 'react'

const MODES = [
  {
    id: 'allowed',
    label: 'Tráfego Normal',
    desc: '100% permitido',
    selected: 'bg-emerald-600 text-white border-emerald-500 shadow-lg shadow-emerald-900/30',
    idle: 'text-gray-500 dark:text-gray-400',
  },
  {
    id: 'mixed',
    label: 'Misto',
    desc: 'proporção livre',
    selected: 'bg-amber-500 text-white border-amber-400 shadow-lg shadow-amber-900/30',
    idle: 'text-gray-500 dark:text-gray-400',
  },
  {
    id: 'attack',
    label: 'Sob Ataque',
    desc: '100% ataques',
    selected: 'bg-red-600 text-white border-red-500 shadow-lg shadow-red-900/30',
    idle: 'text-gray-500 dark:text-gray-400',
  },
]

export default function SimulationConfig({ onApply, applying }) {
  const [mode, setMode] = useState('mixed')
  const [attackRatio, setAttackRatio] = useState(50)

  const allowedPct = 100 - attackRatio

  const handleApply = () => {
    onApply({ mode, attack_ratio: attackRatio / 100 })
  }

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-amber-300/50 dark:border-amber-700/30 p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-brand-text dark:text-gray-200">
            Configuração de Simulação
          </h3>
          <p className="text-xs text-brand-soft dark:text-gray-400 mt-0.5">
            Escolha o tipo de tráfego a ser gerado
          </p>
        </div>
        <span className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full
                         bg-amber-500/15 text-amber-500 border border-amber-500/40 font-bold tracking-wide">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
          SIMULAÇÃO
        </span>
      </div>

      {/* Mode selector */}
      <div className="grid grid-cols-3 gap-2">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`p-3 rounded-xl border text-left transition-all ${
              mode === m.id
                ? m.selected
                : 'bg-brand-bg dark:bg-gray-800 border-brand-muted/20 dark:border-gray-700 hover:bg-brand-primary/5 dark:hover:bg-gray-700 ' + m.idle
            }`}
          >
            <div className="font-semibold text-sm">{m.label}</div>
            <div className={`text-xs mt-0.5 ${mode === m.id ? 'opacity-80' : 'opacity-60'}`}>
              {m.desc}
            </div>
          </button>
        ))}
      </div>

      {/* Slider — só aparece em modo misto */}
      {mode === 'mixed' && (
        <div className="space-y-3">
          <div className="flex justify-between text-xs font-semibold">
            <span className="text-emerald-500">Permitido {allowedPct}%</span>
            <span className="text-red-500">{attackRatio}% Ataques</span>
          </div>

          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={attackRatio}
            onChange={e => setAttackRatio(Number(e.target.value))}
            className="w-full h-2 rounded-full appearance-none cursor-pointer
                       bg-gray-200 dark:bg-gray-700
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-4
                       [&::-webkit-slider-thumb]:h-4
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-white
                       [&::-webkit-slider-thumb]:border-2
                       [&::-webkit-slider-thumb]:border-amber-500
                       [&::-webkit-slider-thumb]:shadow-md"
          />

          {/* Barra visual de proporção */}
          <div className="h-2.5 rounded-full overflow-hidden flex">
            <div
              className="bg-emerald-500 h-full transition-all duration-150"
              style={{ width: `${allowedPct}%` }}
            />
            <div
              className="bg-red-500 h-full transition-all duration-150"
              style={{ width: `${attackRatio}%` }}
            />
          </div>

          <p className="text-xs text-brand-soft dark:text-gray-600 text-center">
            A cada 100 pacotes: ~{allowedPct} permitidos e ~{attackRatio} ataques
          </p>
        </div>
      )}

      {/* Botão aplicar */}
      <button
        onClick={handleApply}
        disabled={applying}
        className="w-full py-2.5 rounded-xl text-sm font-semibold transition-all
                   bg-brand-primary hover:bg-brand-hover text-white
                   disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {applying ? 'Aplicando...' : 'Aplicar Configuração'}
      </button>
    </div>
  )
}
