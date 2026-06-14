const ROWS_PER_PAGE = 10

function tipoBadge(tipo) {
  const t = (tipo || '').toLowerCase()
  if (t.includes('ataque') || t.includes('attack'))
    return 'bg-red-50 dark:bg-red-900/40 text-red-600 dark:text-red-400 border-red-200 dark:border-red-800/50'
  if (t.includes('permitido') || t.includes('allow'))
    return 'bg-emerald-50 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800/50'
  return 'bg-brand-bg dark:bg-gray-800 text-brand-soft dark:text-gray-400 border-brand-muted/30 dark:border-gray-700'
}

export default function LogTable({ data, currentPage, onPageChange, filter, onFilterChange }) {
  const logs = data?.packet_logs?.logs ?? []
  const totalPages = Math.max(1, Math.ceil(logs.length / ROWS_PER_PAGE))
  const safePage = Math.min(currentPage, totalPages)
  const paginated = logs.slice((safePage - 1) * ROWS_PER_PAGE, safePage * ROWS_PER_PAGE)

  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-brand-muted/30 dark:border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-brand-muted/20 dark:border-gray-800 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-brand-text dark:text-gray-200">Detalhes dos Logs</h2>
          <p className="text-xs text-brand-soft dark:text-gray-600 mt-0.5">
            {logs.length > 0 ? `${logs.length} entradas` : 'Sem dados'}
          </p>
        </div>
        <select
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          className="text-sm bg-brand-bg dark:bg-gray-800 border border-brand-muted/30 dark:border-gray-700
                     rounded-lg px-3 py-1.5 text-brand-text dark:text-gray-300
                     focus:outline-none focus:ring-2 focus:ring-brand-primary transition"
        >
          <option value="recentes">Mais Recentes</option>
          <option value="antigos">Mais Antigos</option>
          <option value="todos">Todos</option>
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-brand-bg dark:bg-gray-800/60">
              <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wide text-brand-muted dark:text-gray-400">
                Data e Hora
              </th>
              <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wide text-brand-muted dark:text-gray-400">
                IP Origem
              </th>
              <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wide text-brand-muted dark:text-gray-400">
                IP Destino
              </th>
              <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wide text-brand-muted dark:text-gray-400">
                Tipo
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-brand-muted/10 dark:divide-gray-800">
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-5 py-10 text-center text-sm text-brand-soft dark:text-gray-600">
                  {data ? 'Nenhum log encontrado.' : 'Inicie o monitoramento para ver os logs.'}
                </td>
              </tr>
            ) : (
              paginated.map((log, i) => (
                <tr
                  key={i}
                  className="hover:bg-brand-bg dark:hover:bg-gray-800/30 transition-colors"
                >
                  <td className="px-5 py-3 font-mono text-xs text-brand-soft dark:text-gray-400 whitespace-nowrap">
                    {log.timestamp}
                  </td>
                  <td className="px-5 py-3 font-mono text-xs text-brand-soft dark:text-gray-400 whitespace-nowrap">
                    {log.source_ip}
                  </td>
                  <td className="px-5 py-3 font-mono text-xs text-brand-soft dark:text-gray-400 whitespace-nowrap">
                    {log.destination_ip}
                  </td>
                  <td className="px-5 py-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-md border text-xs font-semibold ${tipoBadge(log.tipo)}`}>
                      {log.tipo}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="px-5 py-4 border-t border-brand-muted/20 dark:border-gray-800 flex items-center justify-between gap-4">
        <span className="text-xs text-brand-soft dark:text-gray-600">
          Página {safePage} de {totalPages}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(Math.max(1, safePage - 1))}
            disabled={safePage <= 1}
            className="px-3 py-1.5 rounded-lg text-sm font-medium
                       bg-brand-bg dark:bg-gray-800 text-brand-text dark:text-gray-400
                       border border-brand-muted/30 dark:border-gray-700
                       hover:bg-brand-primary/10 dark:hover:bg-gray-700 hover:text-brand-primary
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            ← Anterior
          </button>
          <button
            onClick={() => onPageChange(Math.min(totalPages, safePage + 1))}
            disabled={safePage >= totalPages}
            className="px-3 py-1.5 rounded-lg text-sm font-medium
                       bg-brand-bg dark:bg-gray-800 text-brand-text dark:text-gray-400
                       border border-brand-muted/30 dark:border-gray-700
                       hover:bg-brand-primary/10 dark:hover:bg-gray-700 hover:text-brand-primary
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Próxima →
          </button>
        </div>
      </div>
    </div>
  )
}
