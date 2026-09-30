import { Link, useLocation } from 'react-router-dom';
import { Eye, Cpu, Activity, Settings as SettingsIcon } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

export default function Header() {
  const location = useLocation();

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: Eye },
    { path: '/performance', label: 'Performance', icon: Activity },
    { path: '/camera', label: 'Camera', icon: Cpu },
    { path: '/settings', label: 'Settings', icon: SettingsIcon },
  ];

  const isActive = (path: string) => {
    if (path === '/dashboard') return location.pathname === '/dashboard' || location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <header className="bg-white border-b border-border sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-snap-blue rounded-md flex items-center justify-center">
              <Eye className="w-5 h-5 text-white" strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-navy tracking-tight">SnapSight AI</h1>
              <p className="text-xs text-text-secondary -mt-0.5">Edge Vision Intelligence</p>
            </div>
            <div className="ml-4 status-pill bg-green-50 text-green-700 border border-green-200">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              LOCAL PROCESSING
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    'inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors',
                    active
                      ? 'bg-snap-blue/10 text-snap-blue'
                      : 'text-text-secondary hover:text-navy hover:bg-surface-alt',
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}
