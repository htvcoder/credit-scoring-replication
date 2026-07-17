const buildSha = process.env.NEXT_PUBLIC_BUILD_SHA || "local";

export default function VersionPage() {
  return (
    <main className="utility-page">
      <h1>Build version</h1>
      <p>{buildSha}</p>
    </main>
  );
}
