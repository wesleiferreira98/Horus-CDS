import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Bar, Pie, Line } from 'react-chartjs-2'
import useIsDark from '../hooks/useIsDark'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

const MAX_POINTS = 50
const THRESHOLD = 200

function ChartCard({ title, subtitle, children }) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-2xl border border-brand-muted/30 dark:border-gray-800 p-5 flex flex-col">
      <div className="mb-4 shrink-0">
        <h3 className="text-sm font-semibold text-brand-text dark:text-gray-300 uppercase tracking-wide">{title}</h3>
        {subtitle && <p className="text-xs text-brand-soft dark:text-gray-600 mt-0.5">{subtitle}</p>}
      </div>
      <div className="flex-1 min-h-0">
        {children}
      </div>
    </div>
  )
}

function EmptyChart() {
  return (
    <div className="h-52 flex flex-col items-center justify-center gap-2 text-brand-muted dark:text-gray-700">
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
      <span className="text-sm">Aguardando dados...</span>
    </div>
  )
}

export default function ChartsSection({ data, chartType }) {
  const isDark = useIsDark()

  const gridColor = isDark ? '#1f2937' : '#e0eef2'
  const tickColor = isDark ? '#6b7280' : '#96adb6'
  const legendColor = isDark ? '#d1d5db' : '#7c7c84'

  const scaleBase = {
    grid: { color: gridColor },
    ticks: { color: tickColor, font: { size: 11 } },
    border: { display: false },
  }

  const tooltipStyle = {
    backgroundColor: isDark ? '#1f2937' : '#ffffff',
    titleColor: legendColor,
    bodyColor: tickColor,
    borderColor: isDark ? '#374151' : '#96adb6',
    borderWidth: 1,
    padding: 10,
    cornerRadius: 8,
  }

  const pluginBase = {
    legend: { labels: { color: legendColor, font: { size: 11 }, boxWidth: 12, padding: 16 } },
    tooltip: tooltipStyle,
  }

  // Packet logs
  const packetData = {
    labels: ['Ataques', 'Permitidos', 'Inconclusivos'],
    datasets: [{
      label: 'Logs de Rede',
      data: [
        data?.packet_logs?.ataques_detectados ?? 0,
        data?.packet_logs?.requisicoes_permitidas ?? 0,
        data?.packet_logs?.inconclusivos ?? 0,
      ],
      backgroundColor: [
        'rgba(239, 68, 68, 0.75)',
        'rgba(16, 185, 129, 0.75)',
        'rgba(107, 114, 128, 0.75)',
      ],
      borderColor: ['#ef4444', '#10b981', '#6b7280'],
      borderWidth: 1.5,
      borderRadius: chartType === 'bar' ? 6 : 0,
    }],
  }

  const packetOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { ...pluginBase, title: { display: false } },
    ...(chartType === 'bar' && {
      scales: {
        x: scaleBase,
        y: { ...scaleBase, beginAtZero: true },
      },
    }),
  }

  // Predictions
  const resultados = data?.predictions_log?.resultados ?? []
  const normalized = (data?.predictions_log?.normalized_predictions ?? []).slice(-MAX_POINTS)
  const desnormalized = (data?.predictions_log?.desnormalized_predictions ?? []).slice(-MAX_POINTS)
  const labels = resultados.slice(-MAX_POINTS).map((_, i) => `Pct ${i + 1}`)

  const lineOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: pluginBase,
    scales: {
      x: { ...scaleBase, ticks: { ...scaleBase.ticks, maxTicksLimit: 6 } },
      y: { ...scaleBase, beginAtZero: false },
    },
    elements: { point: { radius: 1.5, hoverRadius: 4 } },
  }

  const normalizedData = {
    labels,
    datasets: [{
      label: 'Predição Normalizada',
      data: normalized,
      borderColor: '#1f6e7e',
      backgroundColor: isDark ? 'rgba(31,110,126,0.10)' : 'rgba(31,110,126,0.08)',
      borderWidth: 2,
      fill: true,
      tension: 0.35,
    }],
  }

  const desnormalizedData = {
    labels,
    datasets: [
      {
        label: 'Predição Desnormalizada',
        data: desnormalized,
        borderColor: '#10b981',
        backgroundColor: isDark ? 'rgba(16,185,129,0.08)' : 'rgba(16,185,129,0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.35,
      },
      {
        label: `Limiar (${THRESHOLD})`,
        data: Array(labels.length).fill(THRESHOLD),
        borderColor: 'rgba(239, 68, 68, 0.5)',
        borderWidth: 1.5,
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false,
        tension: 0,
      },
    ],
  }

  const hasData = !!data

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <ChartCard title="Logs de Rede" subtitle="Distribuição de pacotes capturados">
        {!hasData ? <EmptyChart /> : (
          chartType === 'pie'
            ? <Pie data={packetData} options={packetOptions} />
            : <Bar data={packetData} options={packetOptions} />
        )}
      </ChartCard>

      <ChartCard title="Predição Normalizada" subtitle={`Últimos ${MAX_POINTS} pacotes`}>
        {!hasData ? <EmptyChart /> : <Line data={normalizedData} options={lineOptions} />}
      </ChartCard>

      <ChartCard
        title="Predição Desnormalizada"
        subtitle={`Valores brutos · limiar de ataque: ${THRESHOLD}`}
      >
        {!hasData ? <EmptyChart /> : <Line data={desnormalizedData} options={lineOptions} />}
      </ChartCard>
    </div>
  )
}
