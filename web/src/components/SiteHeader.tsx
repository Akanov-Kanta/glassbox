import { REPO_URL } from '../lib/links';
import type { DataSource } from '../lib/types';

interface Props {
  modelSource: DataSource | null;
  trainingSource: DataSource | null;
}

export function SiteHeader({ modelSource, trainingSource }: Props) {
  const demo = modelSource === 'mock' || trainingSource === 'mock';
  const demoText =
    modelSource === 'mock' ? 'demo data' : trainingSource === 'mock' ? 'demo training report' : '';
  return (
    <header className="site-header">
      <div className="wrap site-header__inner">
        <a className="wordmark mono" href="#top" aria-label="Glassbox, back to top">
          <span className="wordmark__mark" aria-hidden="true" />
          glassbox
        </a>
        {demo && (
          <span
            className="demo-badge mono"
            title="The real model files were not found, so this page is running on placeholder data."
          >
            {demoText}
          </span>
        )}
        <nav className="site-nav mono" aria-label="Sections">
          <a href="#playground">playground</a>
          <a href="#how">how it works</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            github
          </a>
        </nav>
      </div>
    </header>
  );
}
