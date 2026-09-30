import { NavLink } from 'react-router-dom';
import { Camera, Mic, FileText, ShieldCheck, BarChart3 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: Parameters<typeof clsx>): string {
  return twMerge(clsx(inputs));
}

export type ActiveTab = 'camera' | 'voice' | 'ocr' | 'safety' | 'metrics';

interface TabBarProps {
  active?: ActiveTab;
  onChange?: (tab: ActiveTab) => void;
}

const tabs: { id: ActiveTab; label: string; icon: React.ComponentType<{ className?: string }>; path?: string }[] = [
  { id: 'camera', label: 'Camera', icon: Camera, path: '/dashboard' },
  { id: 'voice', label: 'Voice', icon: Mic },
  { id: 'ocr', label: 'OCR', icon: FileText },
  { id: 'safety', label: 'Safety', icon: ShieldCheck },
  { id: 'metrics', label: 'Metrics', icon: BarChart3, path: '/performance' },
];

export default function TabBar({ active, onChange }: TabBarProps) {
  return (
    <div className="bg-white border-t border-border md:hidden sticky bottom-0 z-40">
      <div className="grid grid-cols-5">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = active === tab.id;

          const content = (
            <button
            onClick={() => onChange?.(tab.id)}
            className={cn(
              'flex flex-col items-center justify-center gap-1 py-2.5 px-2 transition-colors',
              isActive
                ? 'text-snap-blue'
                : 'text-text-secondary hover:text-navy',
            )}
          >
            <Icon className={cn('w-5 h-5', isActive && 'stroke-[2.5px]')} />
            <span className="text-[11px] font-medium">{tab.label}</span>
          </button>
          );

          if (tab.path && !onChange) {
            return (
              <NavLink
                key={tab.id}
                to={tab.path}
                className={({ isActive: navActive }) =>
                  cn(
                    'flex flex-col items-center justify-center gap-1 py-2.5 px-2 transition-colors',
                    navActive || active === tab.id
                      ? 'text-snap-blue'
                      : 'text-text-secondary hover:text-navy',
                  )
                }
              >
                {({ isActive: navActive }) => (
                <>
                  <Icon className={cn('w-5 h-5', (navActive || active === tab.id) && 'stroke-[2.5px]')} />
                  <span className="text-[11px] font-medium">{tab.label}</span>
                </>
                )}
              </NavLink>
            );
          }

          return <div key={tab.id}>{content}</div>;
        })}
      </div>
    </div>
  );
}
