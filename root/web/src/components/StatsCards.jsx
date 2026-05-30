import { useRef, useEffect } from 'react'

function ShieldAlertIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M12 9v3.75m0-10.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.75c0 5.592 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.57-.598-3.75h-.152c-3.196 0-6.1-1.25-8.25-3.286zm0 13.036h.008v.008H12v-.008z" />
    </svg>
  )
}

function ShieldCheckIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
    </svg>
  )
}

function QuestionIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
    </svg>
  )
}

const CARD_CONFIG = {
  danger: {
    bg: 'bg-red-50 dark:bg-red-950/30',
    border: 'border-red-200 dark:border-red-800/50',
    value: 'text-red-600 dark:text-red-400',
    label: 'text-red-500 dark:text-red-500',
    icon: 'text-red-300 dark:text-red-800',
    upBadge: 'bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400',
    downBadge: 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400',
    Icon: ShieldAlertIcon,
  },
  success: {
    bg: 'bg-emerald-50 dark:bg-emerald-950/30',
    border: 'border-emerald-200 dark:border-emerald-800/50',
    value: 'text-emerald-600 dark:text-emerald-400',
    label: 'text-emerald-500 dark:text-emerald-500',
    icon: 'text-emerald-300 dark:text-emerald-800',
    upBadge: 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400',
    downBadge: 'bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400',
    Icon: ShieldCheckIcon,
  },
  warning: {
    bg: 'bg-amber-50 dark:bg-amber-950/30',
    border: 'border-amber-200 dark:border-amber-800/50',
    value: 'text-amber-600 dark:text-amber-400',
    label: 'text-amber-500 dark:text-amber-500',
    icon: 'text-amber-300 dark:text-amber-800',
    upBadge: 'bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400',
    downBadge: 'bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400',
    Icon: QuestionIcon,
  },
}

const PROGRESS_COLOR = {
  danger:  'bg-red-500 dark:bg-red-400',
  success: 'bg-emerald-500 dark:bg-emerald-400',
  warning: 'bg-amber-500 dark:bg-amber-400',
}

function StatCard({ label, value, total, color }) {
  const prevRef = useRef(value)
  const prevValue = prevRef.current

  useEffect(() => {
    prevRef.current = value
  })

  const diff = value - prevValue
  const pct  = total > 0 ? Math.min((value / total) * 100, 100) : 0
  const c    = CARD_CONFIG[color]
  const { Icon } = c

  return (
    <div className={`relative rounded-2xl border p-5 overflow-hidden transition-colors ${c.bg} ${c.border}`}>
      <Icon className={`absolute right-4 top-4 w-12 h-12 ${c.icon}`} />

      <p className={`text-xs uppercase tracking-widest font-semibold mb-3 pr-14 ${c.label}`}>{label}</p>

      <div className="flex items-end gap-3 relative z-10 pr-16">
        <span className={`text-4xl font-bold tabular-nums ${c.value}`}>
          {value.toLocaleString('pt-BR')}
        </span>
        {diff !== 0 && (
          <span className={`text-xs px-2 py-1 rounded-lg font-bold mb-1 shrink-0 ${diff > 0 ? c.upBadge : c.downBadge}`}>
            {diff > 0 ? '↑' : '↓'} {Math.abs(diff)}
          </span>
        )}
      </div>

      {/* Barra de progresso percentual */}
      {total > 0 && (
        <div className="mt-4 relative z-10">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-500 mb-1.5">
            <span>{pct.toFixed(1)}% do total</span>
            <span>{total.toLocaleString('pt-BR')} total</span>
          </div>
          <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${PROGRESS_COLOR[color]}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default function StatsCards({ data }) {
  const ataques       = data?.packet_logs?.ataques_detectados     ?? 0
  const permitidas    = data?.packet_logs?.requisicoes_permitidas ?? 0
  const inconclusivos = data?.packet_logs?.inconclusivos          ?? 0
  const total         = ataques + permitidas + inconclusivos

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <StatCard label="Ataques Detectados"    value={ataques}       total={total} color="danger"  />
      <StatCard label="Requisições Permitidas" value={permitidas}    total={total} color="success" />
      <StatCard label="Inconclusivos"          value={inconclusivos} total={total} color="warning" />
    </div>
  )
}
