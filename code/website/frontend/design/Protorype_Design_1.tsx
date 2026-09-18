import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Cell, PieChart, Pie
} from 'recharts';
import {
  AlertTriangle, TrendingUp, ShieldAlert, Crosshair,
  Bell, ChevronRight, LogOut, User, Building, Settings, Search,
  Activity, ArrowRight, X, ArrowLeft, Mail, Lock, CheckCircle,
  Briefcase, LineChart as LineChartIcon, Info
} from 'lucide-react';

const MOCK_COMPETITORS = [
  { id: 'c1', name: 'TechCorp Alpha', urgencyScore: 85, trend: 'up', riskLevel: 'High' },
  { id: 'c2', name: 'InnoSystems', urgencyScore: 72, trend: 'stable', riskLevel: 'Medium' },
  { id: 'c3', name: 'GlobalNet Solutions', urgencyScore: 91, trend: 'up', riskLevel: 'Critical' },
  { id: 'c4', name: 'DataSync Pro', urgencyScore: 45, trend: 'down', riskLevel: 'Low' },
];

const FORECAST_DATA = [
  { month: 'Jan', userCompany: 4000, topCompetitor: 2400 },
  { month: 'Feb', userCompany: 3000, topCompetitor: 1398 },
  { month: 'Mar', userCompany: 2000, topCompetitor: 9800 },
  { month: 'Apr', userCompany: 2780, topCompetitor: 3908 },
  { month: 'May', userCompany: 1890, topCompetitor: 4800 },
  { month: 'Jun', userCompany: 2390, topCompetitor: 3800 },
  { month: 'Jul', userCompany: 3490, topCompetitor: 4300 },
];

const RADAR_DATA = [
  { subject: 'Market Share', A: 120, B: 110, fullMark: 150 },
  { subject: 'Innovation', A: 98, B: 130, fullMark: 150 },
  { subject: 'Brand Sentiment', A: 86, B: 130, fullMark: 150 },
  { subject: 'Financial Health', A: 99, B: 100, fullMark: 150 },
  { subject: 'Customer Retention', A: 85, B: 90, fullMark: 150 },
  { subject: 'Agility', A: 65, B: 85, fullMark: 150 },
];

const NEWS_ANOMALIES = [
  { id: 1, title: 'Sudden Drop in TechCorp Alpha Stock', severity: 'high', time: '2 hours ago', source: 'Financial Times', detail: 'Stock plummeted 15% following unconfirmed reports of supply chain disruptions in Asia.' },
  { id: 2, title: 'New Patent Filed by InnoSystems in AI', severity: 'medium', time: '5 hours ago', source: 'TechCrunch', detail: 'Patent covers novel machine learning algorithms for predictive maintenance.' },
  { id: 3, title: 'GlobalNet CEO Resigns Unexpectedly', severity: 'critical', time: '1 day ago', source: 'Bloomberg', detail: 'Board cites "differences in strategic vision." Interim CEO appointed immediately.' },
];

const STRATEGIES = [
  { id: 1, type: 'Aggressive', title: 'Acquisition Spree', risk: 'High', description: 'Target smaller startups in the AI sector to rapidly acquire talent and IP before TechCorp Alpha.', rationale: 'Leverages current cash reserves to outpace competitors in innovation.' },
  { id: 2, type: 'Defensive', title: 'Fortify Core Products', risk: 'Low', description: 'Increase R&D budget by 15% on current top-selling software to prevent churn to GlobalNet.', rationale: 'Secures existing revenue streams against aggressive poaching.' },
  { id: 3, type: 'Strategic', title: 'Partnership with DataSync', risk: 'Medium', description: 'Form a joint venture to share cloud infrastructure costs and undercut market prices.', rationale: 'Reduces operational overhead while expanding market reach.' },
];

const Card = ({ children, className = '', onClick }) => (
  <div
    className={`bg-white rounded-lg border border-neutral-200 shadow-sm overflow-hidden transition-all duration-200 ${onClick ? 'cursor-pointer hover:shadow-md hover:border-neutral-300' : ''} ${className}`}
    onClick={onClick}
  >
    {children}
  </div>
);

const Button = ({ children, variant = 'primary', className = '', ...props }) => {
  const variants = {
    primary: 'bg-black hover:bg-neutral-800 text-white border border-transparent',
    secondary: 'bg-neutral-100 hover:bg-neutral-200 text-black border border-transparent',
    danger: 'bg-red-50 hover:bg-red-100 text-red-600 border border-red-200',
    outline: 'bg-transparent border border-neutral-300 text-black hover:bg-neutral-50'
  };
  return (
    <button
      className={`px-4 py-2 font-semibold rounded-md transition-all flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

const Modal = ({ isOpen, onClose, title, subtitle, children }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col shadow-2xl animate-in zoom-in-95 duration-200">
        <div className="flex justify-between items-start p-6 border-b border-neutral-100">
          <div>
             <h2 className="text-2xl font-bold text-black">{title}</h2>
             {subtitle && <p className="text-neutral-500 mt-1 text-sm">{subtitle}</p>}
          </div>
          <button onClick={onClose} className="text-neutral-400 hover:text-black hover:bg-neutral-100 p-2 rounded-full transition-colors">
            <X size={24} />
          </button>
        </div>
        <div className="p-6 overflow-y-auto flex-1">
          {children}
        </div>
      </div>
    </div>
  );
};

const AuthView = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [persona, setPersona] = useState('investor');

  const handleSubmit = (e) => {
    e.preventDefault();
    onLogin({ email, persona, name: email.split('@')[0] || 'User' });
  };

  return (
    <div className="min-h-screen bg-neutral-50 flex flex-col items-center justify-center p-4 font-sans text-black">
      <div className="mb-8 text-center animate-in slide-in-from-bottom-4 duration-500">
        <img src="Mikir Sekali.png" alt="Mikir Sekali Logo" className="h-24 mx-auto mb-6 object-contain drop-shadow-sm" />
        <h1 className="text-2xl font-bold tracking-tight text-neutral-900">Decision Machine</h1>
        <p className="text-neutral-500 mt-2">Market Intelligence Platform</p>
      </div>

      <Card className="w-full max-w-md p-8 bg-white shadow-xl border-0 animate-in slide-in-from-bottom-8 duration-700">
        <h2 className="text-2xl font-bold text-black mb-6 text-center">
          {isLogin ? 'Welcome Back' : 'Create an Account'}
        </h2>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-neutral-700 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" size={18}/>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-neutral-50 border border-neutral-200 text-black py-2.5 pl-10 pr-4 rounded-md focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent transition-all sm:text-sm"
                placeholder="you@company.com"
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
                <label className="block text-sm font-medium text-neutral-700">Password</label>
                {isLogin && <a href="#" className="text-xs text-neutral-500 hover:text-black">Forgot password?</a>}
            </div>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" size={18}/>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-neutral-50 border border-neutral-200 text-black py-2.5 pl-10 pr-4 rounded-md focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent transition-all sm:text-sm"
                placeholder="••••••••"
              />
            </div>
          </div>

          {!isLogin && (
            <div className="pt-2">
              <label className="block text-sm font-medium text-neutral-700 mb-3">Primary Use Case</label>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {[
                    { id: 'retail', icon: <TrendingUp size={20}/>, label: 'Retail Investor', desc: 'Quick signals' },
                    { id: 'founder', icon: <Building size={20}/>, label: 'Business Owner', desc: 'Strategy focus' },
                    { id: 'manager', icon: <Briefcase size={20}/>, label: 'Fund Manager', desc: 'Deep data' }
                ].map(p => (
                    <button
                        key={p.id}
                        type="button"
                        onClick={() => setPersona(p.id)}
                        className={`p-3 border rounded-lg flex flex-col items-center justify-center gap-1 transition-all text-center ${persona === p.id ? 'bg-neutral-900 text-white border-neutral-900 ring-2 ring-black ring-offset-1' : 'bg-white border-neutral-200 text-neutral-600 hover:border-neutral-300 hover:bg-neutral-50'}`}
                        >
                        {p.icon}
                        <span className="text-xs font-semibold mt-1">{p.label}</span>
                    </button>
                ))}
              </div>
            </div>
          )}

          <div className="pt-4">
            <Button type="submit" className="w-full py-3 text-base">
              {isLogin ? 'Sign In' : 'Create Account'}
            </Button>
          </div>
        </form>

        <div className="mt-8 text-center text-sm">
          {isLogin ? (
             <p className="text-neutral-500">
                Don't have an account? <button onClick={() => setIsLogin(false)} className="text-black font-semibold hover:underline">Sign up</button>
              </p>
          ) : (
            <p className="text-neutral-500">
              Already have an account? <button onClick={() => setIsLogin(true)} className="text-black font-semibold hover:underline">Sign in</button>
            </p>
          )}
        </div>
      </Card>
    </div>
  );
};

const SetupTargetView = ({ onComplete }) => {
  const [targetCompany, setTargetCompany] = useState('');

  return (
    <div className="min-h-screen bg-neutral-900 flex flex-col items-center justify-center p-4 font-sans text-white">
      <div className="w-full max-w-xl p-8 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-white rounded-2xl mb-6 shadow-lg">
          <Crosshair size={32} className="text-black" />
        </div>
        <h2 className="text-3xl font-bold mb-3 tracking-tight">Define Target Vector</h2>
        <p className="text-neutral-400 mb-8 text-lg font-light">
          Enter the primary corporate entity or industry sector for AI analysis.
        </p>

        <form onSubmit={(e) => { e.preventDefault(); onComplete(targetCompany || 'Default Target'); }} className="max-w-md mx-auto">
          <div className="relative mb-6">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-neutral-400" size={20}/>
            <input
              type="text"
              required
              value={targetCompany}
              onChange={(e) => setTargetCompany(e.target.value)}
              className="w-full bg-neutral-800 border border-neutral-700 text-white py-4 pl-12 pr-4 rounded-xl text-lg focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent transition-all placeholder-neutral-500"
              placeholder="e.g., Tesla, AI Sector..."
            />
          </div>
          <button type="submit" className="w-full py-4 text-lg bg-white text-black font-semibold rounded-xl hover:bg-neutral-100 transition-colors flex items-center justify-center gap-2 group">
            Initialize Analysis <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform"/>
          </button>
        </form>
      </div>
    </div>
  );
};

const UrgencyDetailModal = ({ isOpen, onClose }) => (
  <Modal isOpen={isOpen} onClose={onClose} title="Competitor Urgency Analysis" subtitle="Detailed breakdown of threat levels based on market telemetry.">
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {MOCK_COMPETITORS.map(comp => (
          <div key={comp.id} className="bg-neutral-50 p-5 rounded-lg border border-neutral-200">
            <div className="flex justify-between items-start mb-3">
              <div>
                <h3 className="font-bold text-lg text-black">{comp.name}</h3>
                <p className="text-xs text-neutral-500 mt-1">Risk Level: <span className={`font-semibold ${comp.urgencyScore > 80 ? 'text-red-600' : comp.urgencyScore > 50 ? 'text-black' : 'text-green-600'}`}>{comp.riskLevel}</span></p>
              </div>
              <span className={`px-2.5 py-1 text-xs font-bold rounded-full ${comp.urgencyScore > 80 ? 'bg-red-100 text-red-700' : comp.urgencyScore > 50 ? 'bg-neutral-200 text-neutral-800' : 'bg-green-100 text-green-700'}`}>
                Score: {comp.urgencyScore}
              </span>
            </div>
            
            <div className="w-full bg-neutral-200 rounded-full h-2 mb-2 overflow-hidden">
              <div className={`h-full rounded-full ${comp.urgencyScore > 80 ? 'bg-red-500' : 'bg-black'}`} style={{ width: `${comp.urgencyScore}%` }}></div>
            </div>
             <p className="text-xs text-neutral-600 flex items-center gap-1">
                 {comp.trend === 'up' ? <TrendingUp size={14} className="text-red-500"/> : <TrendingUp size={14} className="text-green-500 rotate-180"/>}
                 Trend is {comp.trend === 'up' ? 'increasing' : 'decreasing'}
             </p>
          </div>
        ))}
      </div>

      <div className="bg-white p-6 rounded-lg border border-neutral-200">
        <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
            <Crosshair size={18} className="text-neutral-500"/> Competency Radar Overlay
        </h3>
        <p className="text-sm text-neutral-600 mb-6">Comparing your target vector against the primary competitor across key metrics.</p>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={RADAR_DATA}>
              <PolarGrid stroke="#e5e5e5" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#666', fontSize: 12 }} />
              <Radar name="Target Vector" dataKey="A" stroke="#000" strokeWidth={2} fill="#000" fillOpacity={0.1} />
              <Radar name="Primary Adversary" dataKey="B" stroke="#ef4444" strokeWidth={2} fill="#ef4444" fillOpacity={0.1} />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}/>
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  </Modal>
);

const ForecastDetailModal = ({ isOpen, onClose }) => (
  <Modal isOpen={isOpen} onClose={onClose} title="Battle Forecast & Projections" subtitle="Algorithmic projections comparing trajectory against primary adversary.">
    <div className="space-y-6">
      
      <div className="bg-white p-4 rounded-lg border border-neutral-200 h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={FORECAST_DATA} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
            <XAxis dataKey="month" stroke="#a3a3a3" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
            <YAxis stroke="#a3a3a3" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
            <RechartsTooltip 
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e5e5', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} 
              itemStyle={{ fontWeight: '500' }}
            />
            <Legend wrapperStyle={{fontSize: '12px', paddingTop: '20px'}}/>
            <Line type="monotone" dataKey="userCompany" name="Target Trajectory" stroke="#000" strokeWidth={3} dot={{r: 4, fill: '#000', strokeWidth: 0}} activeDot={{ r: 6 }} />
            <Line type="monotone" dataKey="topCompetitor" name="Adversary Projection" stroke="#ef4444" strokeWidth={3} strokeDasharray="5 5" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-100 flex flex-col justify-center items-center text-center">
          <h4 className="text-neutral-500 font-medium text-xs uppercase tracking-wider mb-1">Market Delta</h4>
          <p className="text-3xl font-bold text-green-600">+4.5%</p>
          <p className="text-[10px] text-neutral-400 mt-1">vs Last Quarter</p>
        </div>
        <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-100 flex flex-col justify-center items-center text-center">
          <h4 className="text-neutral-500 font-medium text-xs uppercase tracking-wider mb-1">Revenue Risk exposure</h4>
          <p className="text-3xl font-bold text-red-600">-$1.2M</p>
           <p className="text-[10px] text-neutral-400 mt-1">Estimated 6mo impact</p>
        </div>
        <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-100 flex flex-col justify-center items-center text-center">
          <h4 className="text-neutral-500 font-medium text-xs uppercase tracking-wider mb-1">Win Probability</h4>
          <p className="text-3xl font-bold text-black">68%</p>
           <p className="text-[10px] text-neutral-400 mt-1">Current strategy</p>
        </div>
      </div>
    </div>
  </Modal>
);

const StrategyDetailModal = ({ isOpen, onClose, persona }) => (
  <Modal isOpen={isOpen} onClose={onClose} title="Strategy Advisory & Risk Management" subtitle="Actionable imperatives synthesized from anomaly detection and urgency scoring.">
    <div className="space-y-6">
      
      {/* Dynamic Summary based on Persona */}
      <div className="bg-blue-50/50 p-4 rounded-lg border border-blue-100 flex gap-3">
         <Info className="text-blue-500 flex-shrink-0" size={20} />
         <div>
             <h4 className="font-semibold text-blue-900 text-sm mb-1">AI Executive Summary</h4>
             <p className="text-sm text-blue-800 leading-relaxed">
                 {persona === 'retail' && "Hold steady. Recent anomalies indicate short-term volatility, but core fundamentals remain stronger than top competitors. Avoid panic selling."}
                 {persona === 'founder' && "Focus on mitigating risk from GlobalNet's aggressive moves. Reallocate Q3 budget towards fortifying core product features to ensure customer retention."}
                 {persona === 'manager' && "Fundamental data supports current valuation, however, noise from competitor patents requires monitoring. Maintain position; set alerts for specific technical indicators."}
                 {!['retail', 'founder', 'manager'].includes(persona) && "Review recommended strategies to mitigate identified competitor risks and capitalize on market deltas."}
             </p>
         </div>
      </div>

      <div className="space-y-4">
        <h3 className="font-bold text-lg border-b pb-2">Recommended Protocols</h3>
        {STRATEGIES.map(strategy => (
          <div key={strategy.id} className="bg-white p-5 rounded-lg border border-neutral-200 relative overflow-hidden transition-all hover:border-neutral-300 hover:shadow-sm">
            <div className={`absolute left-0 top-0 bottom-0 w-1.5 ${
                strategy.risk === 'High' ? 'bg-red-500' :
                strategy.risk === 'Medium' ? 'bg-yellow-500' :
                'bg-green-500'
              }`}></div>
            <div className="pl-4">
                <div className="flex justify-between items-start mb-2">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold uppercase tracking-wider text-neutral-500">{strategy.type}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            strategy.risk === 'High' ? 'bg-red-100 text-red-700' :
                            strategy.risk === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-700'
                        }`}>Risk: {strategy.risk}</span>
                    </div>
                    <h3 className="text-lg font-bold text-black">{strategy.title}</h3>
                </div>
                </div>
                <p className="text-sm text-neutral-700 mb-3">{strategy.description}</p>
                <div className="bg-neutral-50 p-3 rounded text-sm text-neutral-600 border border-neutral-100">
                    <span className="font-semibold text-black">Rationale:</span> {strategy.rationale}
                </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  </Modal>
);

const Dashboard = ({ user, target, onLogout }) => {
  const [activeModal, setActiveModal] = useState(null);
  const [showNotifications, setShowNotifications] = useState(false);
  const [unreadCount, setUnreadCount] = useState(NEWS_ANOMALIES.length);

  const handleNotificationClick = () => {
    setShowNotifications(!showNotifications);
    setUnreadCount(0);
  };

  // Helper to determine greeting based on persona
  const getPersonaGreeting = () => {
      switch(user.persona) {
          case 'retail': return "Investor Overview";
          case 'founder': return "Executive Dashboard";
          case 'manager': return "Fund Manager Terminal";
          default: return "System Overview";
      }
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans">
      
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white border-b border-neutral-200 px-4 md:px-8 py-3 flex justify-between items-center shadow-sm">
        <div className="flex items-center gap-4">
          <img src="Mikir Sekali.png" alt="Mikir Sekali Logo" className="h-10 object-contain" />
          <div className="hidden sm:block pl-4 border-l border-neutral-200">
            <h1 className="text-xl font-bold text-black tracking-tight leading-none">MikirSekali</h1>
            <p className="text-xs font-medium text-neutral-500 mt-1">Target: <span className="text-black font-semibold">{target}</span></p>
          </div>
        </div>

        <div className="flex items-center gap-4 sm:gap-6">
          {/* Notifications */}
          <div className="relative">
            <button 
              onClick={handleNotificationClick}
              className="p-2 rounded-full hover:bg-neutral-100 transition-colors relative"
            >
              <Bell size={20} className="text-neutral-600" />
              {unreadCount > 0 && (
                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white"></span>
              )}
            </button>

            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 sm:w-96 bg-white rounded-xl shadow-xl border border-neutral-200 overflow-hidden z-50 animate-in slide-in-from-top-2 duration-200">
                <div className="p-4 border-b border-neutral-100 flex justify-between items-center bg-neutral-50/50">
                  <h3 className="font-semibold text-black flex items-center gap-2 text-sm">
                    <AlertTriangle size={16} className="text-red-500"/> Market Anomalies
                  </h3>
                  <span className="text-xs font-medium text-neutral-500">{NEWS_ANOMALIES.length} Alerts</span>
                </div>
                <div className="max-h-[350px] overflow-y-auto">
                  {NEWS_ANOMALIES.map(news => (
                    <div key={news.id} className="p-4 border-b border-neutral-50 hover:bg-neutral-50 transition-colors cursor-pointer group">
                      <div className="flex items-start gap-3">
                        <div className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${news.severity === 'critical' ? 'bg-red-500' : news.severity === 'high' ? 'bg-black' : 'bg-yellow-500'}`} />
                        <div>
                          <p className="text-sm font-semibold text-black leading-tight group-hover:text-neutral-700 transition-colors">{news.title}</p>
                          <p className="text-xs text-neutral-500 mt-1 line-clamp-2">{news.detail}</p>
                          <div className="flex items-center gap-2 mt-2 text-[10px] font-medium text-neutral-400">
                            <span>{news.source}</span>
                            <span>•</span>
                            <span>{news.time}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="p-3 text-center bg-neutral-50 hover:bg-neutral-100 cursor-pointer text-xs font-semibold text-black transition-colors border-t border-neutral-100">
                  View All Alerts
                </div>
              </div>
            )}
          </div>

          <div className="h-8 w-px bg-neutral-200 hidden sm:block"></div>
          
          {/* User Profile */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:block text-right">
              <p className="text-sm font-semibold text-black leading-tight">{user.name}</p>
              <p className="text-xs text-neutral-500 capitalize">{user.persona}</p>
            </div>
            <div className="w-8 h-8 rounded-full bg-neutral-100 border border-neutral-200 flex items-center justify-center text-neutral-600">
              <User size={16} />
            </div>
            <button onClick={onLogout} className="ml-1 p-2 text-neutral-400 hover:text-red-600 rounded-full hover:bg-red-50 transition-colors" title="Logout">
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-4 md:p-8 max-w-7xl mx-auto space-y-6">
        
        {/* Overview Banner */}
        <div className="bg-black text-white rounded-xl p-6 md:p-8 shadow-md flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h2 className="text-2xl md:text-3xl font-bold mb-2 tracking-tight">{getPersonaGreeting()}</h2>
            <p className="text-neutral-400 text-sm md:text-base">
              Real-time intelligence aggregation active for <span className="text-white font-semibold">{target}</span>.
            </p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-white/10 rounded-lg border border-white/20 backdrop-blur-sm">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
            <span className="text-sm font-medium">Live Sync</span>
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Urgency Competitor Card */}
          <Card className="flex flex-col group h-[400px]" onClick={() => setActiveModal('urgency')}>
            <div className="p-5 border-b border-neutral-100 flex justify-between items-center bg-white group-hover:bg-neutral-50/50 transition-colors">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-50 rounded-lg">
                    <ShieldAlert className="text-red-600" size={24} />
                </div>
                <h2 className="text-lg font-bold text-black">Urgency Competitor</h2>
              </div>
              <ChevronRight className="text-neutral-400 group-hover:text-black transition-colors" size={20}/>
            </div>
            <div className="p-5 flex-1 flex flex-col justify-between">
                
                {/* Simplified view for Persona 1, detailed for others */}
                {user.persona === 'retail' ? (
                     <div className="text-center h-full flex flex-col justify-center items-center">
                         <AlertTriangle size={48} className="text-yellow-500 mb-4 opacity-80" />
                         <p className="text-neutral-600 text-sm max-w-xs">
                             Competitor activity is elevated. <span className="font-semibold text-black">GlobalNet Solutions</span> is showing aggressive upward trends.
                         </p>
                     </div>
                ) : (
                    <div className="space-y-4">
                        {MOCK_COMPETITORS.slice(0,3).map((comp, i) => (
                        <div key={comp.id} className="flex flex-col gap-1">
                            <div className="flex justify-between items-end">
                                <span className="font-semibold text-sm text-black">{i+1}. {comp.name}</span>
                                <span className={`text-xs font-medium ${comp.trend==='up' ? 'text-red-500' : 'text-green-500'}`}>
                                    {comp.trend === 'up' ? 'Threat ↑' : 'Stable'}
                                </span>
                            </div>
                            <div className="w-full h-1.5 bg-neutral-100 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full ${comp.urgencyScore > 80 ? 'bg-red-500' : 'bg-black'}`} style={{width: `${comp.urgencyScore}%`}}></div>
                            </div>
                        </div>
                        ))}
                    </div>
                )}
              
              <div className="mt-4 pt-4 border-t border-neutral-100 text-sm text-neutral-500 flex justify-between items-center">
                  <span>Based on multi-vector analysis</span>
                  <span className="font-medium text-black group-hover:underline">View details</span>
              </div>
            </div>
          </Card>

          {/* Battle Forecast Card */}
          <Card className="flex flex-col group h-[400px]" onClick={() => setActiveModal('forecast')}>
            <div className="p-5 border-b border-neutral-100 flex justify-between items-center bg-white group-hover:bg-neutral-50/50 transition-colors">
               <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-50 rounded-lg">
                    <LineChartIcon className="text-blue-600" size={24} />
                </div>
                <h2 className="text-lg font-bold text-black">Battle Forecast</h2>
              </div>
              <ChevronRight className="text-neutral-400 group-hover:text-black transition-colors" size={20}/>
            </div>
            <div className="p-5 flex-1 flex flex-col relative">
                <p className="text-xs text-neutral-500 mb-4">6-Month Trajectory Projection vs Top Competitor</p>
                <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={FORECAST_DATA}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                            <Line type="monotone" dataKey="userCompany" stroke="#000" strokeWidth={3} dot={false} />
                            <Line type="monotone" dataKey="topCompetitor" stroke="#ef4444" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="mt-4 pt-4 border-t border-neutral-100 text-sm text-neutral-500 flex justify-between items-center">
                  <span>Historical & Predictive</span>
                  <span className="font-medium text-black group-hover:underline">Open charts</span>
              </div>
            </div>
          </Card>

        </div>

        {/* Strategy Advise Card (Full Width) */}
        <Card className="group overflow-hidden" onClick={() => setActiveModal('strategy')}>
            <div className="p-6 border-b border-neutral-100 flex justify-between items-center bg-white group-hover:bg-neutral-50/50 transition-colors">
               <div className="flex items-center gap-3">
                <div className="p-2 bg-green-50 rounded-lg">
                    <Activity className="text-green-600" size={24}/>
                </div>
                <div>
                    <h2 className="text-lg font-bold text-black">Strategy Advisory & Risk Management</h2>
                    <p className="text-xs text-neutral-500 mt-0.5">AI-generated action plans</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm font-medium text-neutral-500 group-hover:text-black transition-colors">
                <span>View Full Report</span>
                <ChevronRight size={18}/>
              </div>
            </div>
            
            <div className="p-6 bg-white">
               {/* Show concise advice for retail, full list for others */}
               {user.persona === 'retail' ? (
                   <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-100 flex items-start gap-4">
                       <Info className="text-neutral-400 mt-0.5" size={20}/>
                       <div>
                           <h4 className="font-semibold text-black text-sm mb-1">Market Sentiment: Hold</h4>
                           <p className="text-sm text-neutral-600">Current volatility suggests holding positions. Short-term dips are expected due to competitor news, but long-term indicators remain stable.</p>
                       </div>
                   </div>
               ) : (
                   <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {STRATEGIES.map((strat, i) => (
                        <div key={i} className="p-4 rounded-lg border border-neutral-100 bg-neutral-50 hover:bg-white hover:shadow-sm transition-all hover:border-neutral-200">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">{strat.type}</span>
                            <span className={`w-2 h-2 rounded-full ${i===0 ? 'bg-red-500' : i===1 ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
                        </div>
                        <h3 className="font-bold text-black mb-1">{strat.title}</h3>
                        <p className="text-sm text-neutral-600 line-clamp-2">{strat.description}</p>
                        </div>
                    ))}
                    </div>
               )}
            </div>
        </Card>

      </main>

      {/* Modals */}
      <UrgencyDetailModal isOpen={activeModal === 'urgency'} onClose={() => setActiveModal(null)} />
      <ForecastDetailModal isOpen={activeModal === 'forecast'} onClose={() => setActiveModal(null)} />
      <StrategyDetailModal isOpen={activeModal === 'strategy'} onClose={() => setActiveModal(null)} persona={user.persona} />

    </div>
  );
};

export default function App() {
  const [user, setUser] = useState(null);
  const [targetConfigured, setTargetConfigured] = useState(null);

  const handleLogin = (userData) => setUser(userData);
  const handleSetupComplete = (target) => setTargetConfigured(target);
  const handleLogout = () => { setUser(null); setTargetConfigured(null); };

  if (!user) return <AuthView onLogin={handleLogin} />;
  if (!targetConfigured) return <SetupTargetView onComplete={handleSetupComplete} />;

  return <Dashboard user={user} target={targetConfigured} onLogout={handleLogout} />;
}