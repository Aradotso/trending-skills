```markdown
---
name: mike-ai-legal-platform
description: OSS AI legal platform with Next.js frontend and Express backend for document processing and legal workflows
triggers:
  - set up mike legal platform
  - add mike to my project
  - configure mike backend
  - mike document processing
  - mike supabase integration
  - mike frontend setup
  - deploy mike legal AI
  - mike API endpoints
---

# Mike AI Legal Platform

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Mike is an open-source AI legal platform consisting of a Next.js frontend and an Express backend. It supports document processing (including DOC/DOCX to PDF via LibreOffice), Supabase Auth and Postgres, S3-compatible object storage, and multiple AI model providers.

---

## Project Structure

```
mike/
├── frontend/          # Next.js application
├── backend/           # Express API, Supabase access, document processing
│   └── migrations/
│       └── 000_one_shot_schema.sql  # Fresh DB schema
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/willchen96/mike.git
cd mike

# Install dependencies for both services
npm install --prefix backend
npm install --prefix frontend

# Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.local.example frontend/.env.local
```

### Database Setup

Run the one-shot schema in the Supabase SQL editor:

```
backend/migrations/000_one_shot_schema.sql
```

---

## Environment Configuration

### Backend (`backend/.env`)

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=$SUPABASE_SERVICE_ROLE_KEY

# S3-compatible storage (e.g. Cloudflare R2)
S3_ENDPOINT=$S3_ENDPOINT
S3_BUCKET=$S3_BUCKET
S3_ACCESS_KEY_ID=$S3_ACCESS_KEY_ID
S3_SECRET_ACCESS_KEY=$S3_SECRET_ACCESS_KEY
S3_REGION=$S3_REGION

# AI Model Provider(s) — enable at least one
OPENAI_API_KEY=$OPENAI_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

# Server
PORT=4000
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=$NEXT_PUBLIC_SUPABASE_ANON_KEY
NEXT_PUBLIC_API_URL=http://localhost:4000
```

---

## Running the Platform

```bash
# Start backend (Express API on port 4000)
npm run dev --prefix backend

# Start frontend (Next.js on port 3000)
npm run dev --prefix frontend
```

Open `http://localhost:3000`.

---

## Key Commands

```bash
# Build checks
npm run build --prefix backend
npm run build --prefix frontend

# Lint
npm run lint --prefix frontend
```

---

## Required Services

| Service | Purpose |
|---|---|
| Supabase | Auth + Postgres database |
| S3-compatible storage | Document/file storage (e.g. Cloudflare R2) |
| AI model provider | At least one (OpenAI, Anthropic, etc.) |
| LibreOffice | DOC/DOCX → PDF conversion |

### Installing LibreOffice (Ubuntu/Debian)

```bash
sudo apt-get install -y libreoffice
```

---

## Common Patterns

### Supabase Client (Backend)

```typescript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Fetch documents for a user
const { data, error } = await supabase
  .from('documents')
  .select('*')
  .eq('user_id', userId);
```

### Supabase Auth (Frontend)

```typescript
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

const supabase = createClientComponentClient();

// Sign in
const { data, error } = await supabase.auth.signInWithPassword({
  email: process.env.NEXT_PUBLIC_USER_EMAIL,
  password: process.env.NEXT_PUBLIC_USER_PASSWORD,
});

// Get current session
const { data: { session } } = await supabase.auth.getSession();
```

### Uploading a Document (Frontend → Backend)

```typescript
// frontend/lib/api.ts
export async function uploadDocument(file: File, token: string) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/documents/upload`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```

### Express Route with Auth Middleware (Backend)

```typescript
// backend/src/routes/documents.ts
import { Router, Request, Response } from 'express';
import { requireAuth } from '../middleware/auth';
import { supabase } from '../lib/supabase';

const router = Router();

router.get('/', requireAuth, async (req: Request, res: Response) => {
  const userId = (req as any).user.id;

  const { data, error } = await supabase
    .from('documents')
    .select('id, name, created_at, status')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });

  if (error) return res.status(500).json({ error: error.message });
  return res.json(data);
});

export default router;
```

### S3 File Upload (Backend)

```typescript
// backend/src/lib/storage.ts
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { randomUUID } from 'crypto';

const s3 = new S3Client({
  endpoint: process.env.S3_ENDPOINT,
  region: process.env.S3_REGION,
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY_ID!,
    secretAccessKey: process.env.S3_SECRET_ACCESS_KEY!,
  },
});

export async function uploadToStorage(
  buffer: Buffer,
  mimeType: string,
  originalName: string
): Promise<string> {
  const key = `documents/${randomUUID()}/${originalName}`;

  await s3.send(
    new PutObjectCommand({
      Bucket: process.env.S3_BUCKET!,
      Key: key,
      Body: buffer,
      ContentType: mimeType,
    })
  );

  return key;
}
```

### DOC/DOCX to PDF Conversion (Backend)

```typescript
// backend/src/lib/convert.ts
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';

const execAsync = promisify(exec);

export async function convertDocToPdf(inputPath: string): Promise<string> {
  const dir = path.dirname(inputPath);

  await execAsync(
    `libreoffice --headless --convert-to pdf --outdir ${dir} ${inputPath}`
  );

  const baseName = path.basename(inputPath, path.extname(inputPath));
  const outputPath = path.join(dir, `${baseName}.pdf`);

  // Verify output exists
  await fs.access(outputPath);
  return outputPath;
}
```

### Calling an AI Model (Backend)

```typescript
// backend/src/lib/ai.ts
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function summarizeDocument(text: string): Promise<string> {
  const response = await openai.chat.completions.create({
    model: 'gpt-4o',
    messages: [
      {
        role: 'system',
        content: 'You are a legal assistant. Summarize the following document concisely.',
      },
      { role: 'user', content: text },
    ],
  });

  return response.choices[0].message.content ?? '';
}
```

---

## Next.js API Route Pattern (Frontend)

```typescript
// frontend/app/api/documents/route.ts
import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';

export async function GET() {
  const supabase = createRouteHandlerClient({ cookies });

  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/documents`, {
    headers: { Authorization: `Bearer ${session.access_token}` },
  });

  const data = await res.json();
  return NextResponse.json(data);
}
```

---

## Troubleshooting

### LibreOffice not found
```bash
which libreoffice
# If missing:
sudo apt-get install -y libreoffice   # Debian/Ubuntu
brew install --cask libreoffice       # macOS
```

### Supabase connection errors
- Verify `SUPABASE_URL` includes `https://` prefix.
- Service role key must be used on the backend (never expose it in the frontend).
- Run `000_one_shot_schema.sql` via the Supabase SQL editor before first launch.

### S3 upload failures with Cloudflare R2
- Set `S3_REGION` to `auto` for R2.
- Ensure `S3_ENDPOINT` is the full R2 endpoint: `https://<account-id>.r2.cloudflarestorage.com`.

### Port conflicts
- Backend defaults to port `4000`. Override with `PORT=XXXX` in `backend/.env`.
- Frontend defaults to port `3000`. Override with `next dev -p XXXX`.

### Build errors
```bash
# Check TypeScript errors
npm run build --prefix backend
npm run build --prefix frontend

# Lint issues
npm run lint --prefix frontend
```

### CORS errors between frontend and backend
Add the frontend origin to your Express CORS config:

```typescript
import cors from 'cors';
app.use(cors({ origin: process.env.FRONTEND_URL || 'http://localhost:3000' }));
```
```
