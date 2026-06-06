import type { ReactNode } from 'react';
import { Header } from './Header';
import { Footer } from './Footer';

interface LayoutProps {
  children: ReactNode;
  showNav?: boolean;
  showBackButton?: boolean;
  backTo?: string;
  backLabel?: string;
  hideFooter?: boolean;
  fullHeight?: boolean;
}

export function Layout({
  children,
  showNav = true,
  showBackButton = false,
  backTo,
  backLabel,
  hideFooter = false,
  fullHeight = false
}: LayoutProps) {
  return (
    <div className={`min-h-screen flex flex-col ${fullHeight ? 'h-screen overflow-hidden' : ''}`}>
      <Header
        showNav={showNav}
        showBackButton={showBackButton}
        backTo={backTo}
        backLabel={backLabel}
      />
      <main className={`flex-grow ${fullHeight ? 'overflow-hidden' : ''}`}>
        {children}
      </main>
      {!hideFooter && <Footer />}
    </div>
  );
}
