import type { NextRequest } from "next/server";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function proxy(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const search = new URL(request.url).search;
  const target = `http://127.0.0.1:8788/api/${path.join("/")}${search}`;
  const body =
    request.method === "GET" || request.method === "HEAD"
      ? undefined
      : await request.text();
  const response = await fetch(target, {
    method: request.method,
    headers: {
      "Content-Type": request.headers.get("Content-Type") ?? "application/json",
      ...(path[0] === "desktop-execution" ? {
        "Origin": request.headers.get("Origin") ?? "",
        "X-MoneyMore-Desktop": request.headers.get("X-MoneyMore-Desktop") ?? "",
      } : {}),
    },
    body,
    cache: "no-store",
  });
  return new Response(response.body, {
    status: response.status,
    headers: {
      "Content-Type":
        response.headers.get("Content-Type") ?? "application/json; charset=utf-8",
    },
  });
}

export async function GET(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxy(request, context);
}
