import { Hero } from './components/Hero';
import { HowItWorks } from './components/HowItWorks';
import { Playground } from './components/Playground';
import { SiteHeader } from './components/SiteHeader';
import { useModelLoader } from './hooks/useModelLoader';
import { AUTHOR, PORTFOLIO_URL, REPO_URL } from './lib/links';

export default function App() {
  const { state, retry } = useModelLoader();
  const preview = state.preview;
  const app = state.status === 'ready' ? state.app : null;

  return (
    <>
      <a className="skip-link" href="#playground">
        Skip to the playground
      </a>
      <SiteHeader
        modelSource={app?.modelSource ?? preview?.modelSource ?? null}
        trainingSource={app?.trainingSource ?? preview?.trainingSource ?? null}
      />
      <main>
        <Hero preview={preview} />
        <Playground state={state} onRetry={retry} />
        <HowItWorks preview={preview} weightBytes={app?.weightBytes ?? null} />
      </main>
      <footer className="site-footer">
        <div className="wrap site-footer__inner mono">
          <p>
            Built by{' '}
            <a href={PORTFOLIO_URL} target="_blank" rel="noreferrer">
              {AUTHOR}
            </a>
            .
          </p>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            github.com/Akanov-Kanta/glassbox
          </a>
        </div>
      </footer>
    </>
  );
}
