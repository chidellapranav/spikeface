const $ = id => document.getElementById(id);
let result, previewUrl;
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
api('/api/model').then(data => {
  $('model').textContent = data.metadata.verification_fixture
    ? 'Verification fixture loaded. Replay checks the committed trace. Image predictions are exploratory; no held-out accuracy has been measured.'
    : `Model loaded: ${data.metadata.name || 'Custom weights'}. Accuracy must be established on a held-out dataset.`;
  $('fixture').hidden = !data.metadata.verification_fixture;
}).catch(error => { $('model').textContent = error.message; });
$('image').addEventListener('change', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = $('image').files[0];
  $('preview').hidden = !file;
  if (file) { previewUrl = URL.createObjectURL(file); $('preview').src = previewUrl; }
});
async function run(fixture) {
  $('run').disabled = $('fixture').disabled = true;
  $('result').setAttribute('aria-busy', 'true');
  $('state').textContent = 'Running'; $('status').textContent = 'Processing spike frames…';
  $('output').hidden = true;
  try {
    let payload = {fixture};
    if (!fixture) {
      const file = $('image').files[0];
      if (!file) throw new Error('Choose a face image first.');
      if (file.size > 4 * 1024 * 1024) throw new Error('Choose an image smaller than 4 MiB.');
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Could not read this image.')); reader.readAsDataURL(file);
      });
      payload = {image: dataUrl.split(',')[1], timesteps: Number($('timesteps').value), seed: Number($('seed').value)};
    }
    result = await api('/api/infer', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    $('identity').textContent = result.identity;
    $('time').textContent = `${result.seconds.toFixed(3)} s`;
    $('steps').textContent = result.timesteps;
    $('scores').replaceChildren(...result.top5.map(item => {
      const li = document.createElement('li'), label = document.createElement('span'), score = document.createElement('strong');
      label.textContent = item.label; score.textContent = item.score.toFixed(6); li.append(label, score); return li;
    }));
    $('sparsity').textContent = Object.entries(result.sparsity).map(([name, value]) => `${name}: ${(value * 100).toFixed(1)}%`).join(' · ');
    $('saturation').textContent = result.score_at_limit ? 'A final accumulator is at its limit. Saturation may affect ranking.' : '';
    $('status').textContent = `Completed with ${result.backend}.`;
    $('state').textContent = 'Complete'; $('output').hidden = false;
  } catch (error) { $('status').textContent = error.message; $('state').textContent = 'Could not run'; }
  finally { $('run').disabled = $('fixture').disabled = false; $('result').setAttribute('aria-busy','false'); }
}
$('infer').addEventListener('submit', event => { event.preventDefault(); run(false); });
$('fixture').addEventListener('click', () => run(true));
$('download').addEventListener('click', () => {
  if (!result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'spikeface-result.json'; a.click(); URL.revokeObjectURL(url);
});
