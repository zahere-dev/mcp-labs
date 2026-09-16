"""
Streamable HTTP MCP server exposing a deliberately large, multi-domain
toolset (CRM, ticketing, files, weather, math, inventory, billing, HR,
analytics, notifications, calendar, project management -- 31 tools total).
The full set of tool schemas alone is large enough (~5K tokens) to make
progressive discovery worth it even for a single server; the point of this
lab is not the tools themselves -- it's the *client's* job to decide whether
to dump every one of these schemas into the model's context up front, or to
let the model discover them progressively (see client_with_keyword_search.py,
client_with_embeddings_search.py, client_langchain.py).

Run:
    python server.py
Serves streamable-http at http://127.0.0.1:8300/mcp
"""

import logging
import time
import uuid
from typing import Literal

from mcp import types
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("progressive-discovery-server")

mcp = FastMCP(
    name="Progressive Discovery Demo Server",
    host="127.0.0.1",
    port=8300,
    stateless_http=True,
)

# ---------------------------------------------------------------------------
# 2026-07-28 spec: tools/list (and prompts/list, resources/list,
# resources/read, server/discover) responses carry `ttlMs` / `cacheScope`
# cache hints. The Python SDK doesn't model these as first-class fields yet,
# so we attach them via the standard `_meta` extension point instead --
# exactly the mechanism `_meta` exists for. This tool catalog barely ever
# changes at runtime, so a generous TTL is reasonable; a server backed by a
# fast-moving plugin registry would want a much shorter one.
# ---------------------------------------------------------------------------
TOOLS_LIST_TTL_MS = 5 * 60 * 1000
TOOLS_LIST_CACHE_SCOPE = "session"

# ---------------------------------------------------------------------------
# In-memory "databases" so the tools do something real and inspectable.
# ---------------------------------------------------------------------------
_contacts: dict[str, dict] = {}
_issues: dict[str, dict] = {}
_files: dict[str, str] = {
    "/notes/todo.txt": "1. Ship progressive discovery demo\n2. Write tests",
}
_inventory: dict[str, dict] = {
    "WIDGET-100": {"sku": "WIDGET-100", "quantity": 250, "warehouse": "east-1"},
}
_invoices: dict[str, dict] = {}
_employees: dict[str, dict] = {}
_time_off_requests: dict[str, dict] = {}
_reports: dict[str, dict] = {}
_events: dict[str, dict] = {}
_tasks: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# CRM tools
# ---------------------------------------------------------------------------


@mcp.tool()
def crm_create_contact(name: str, email: str, company: str = "") -> dict:
    """Create a new CRM contact record and return its generated id. The
    email is used as the primary key for later deduplication when contacts
    are imported in bulk from an external source, so it should be the
    contact's canonical work email where one is known."""
    contact_id = str(uuid.uuid4())[:8]
    _contacts[contact_id] = {"id": contact_id, "name": name, "email": email, "company": company}
    logger.info("crm_create_contact -> %s", contact_id)
    return _contacts[contact_id]


@mcp.tool()
def crm_update_contact(contact_id: str, field: str, value: str) -> dict:
    """Update a single field (name, email, or company) on an existing CRM contact."""
    if contact_id not in _contacts:
        raise ValueError(f"No contact with id {contact_id}")
    if field not in {"name", "email", "company"}:
        raise ValueError("field must be one of: name, email, company")
    _contacts[contact_id][field] = value
    return _contacts[contact_id]


@mcp.tool()
def crm_search_contacts(query: str) -> list[dict]:
    """Search CRM contacts by substring match against name, email, or company."""
    q = query.lower()
    return [c for c in _contacts.values() if q in " ".join(c.values()).lower()]


# ---------------------------------------------------------------------------
# Ticketing tools
# ---------------------------------------------------------------------------


@mcp.tool()
def ticketing_create_issue(title: str, body: str = "", priority: Literal["low", "medium", "high"] = "medium") -> dict:
    """File a new support/engineering ticket and return its id. High
    priority tickets are routed to the on-call rotation immediately, while
    low and medium priority tickets are triaged during normal business
    hours by whichever team owns the affected component."""
    issue_id = f"ISSUE-{len(_issues) + 1}"
    _issues[issue_id] = {
        "id": issue_id,
        "title": title,
        "body": body,
        "priority": priority,
        "status": "open",
        "created_at": time.time(),
    }
    logger.info("ticketing_create_issue -> %s", issue_id)
    return _issues[issue_id]


@mcp.tool()
def ticketing_list_issues(status: Literal["open", "closed", "all"] = "open") -> list[dict]:
    """List tickets, optionally filtered by status."""
    if status == "all":
        return list(_issues.values())
    return [i for i in _issues.values() if i["status"] == status]


@mcp.tool()
def ticketing_close_issue(issue_id: str, resolution: str = "") -> dict:
    """Close a ticket by id with an optional resolution note."""
    if issue_id not in _issues:
        raise ValueError(f"No issue with id {issue_id}")
    _issues[issue_id]["status"] = "closed"
    _issues[issue_id]["resolution"] = resolution
    return _issues[issue_id]


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


@mcp.tool()
def files_read_file(path: str) -> str:
    """Read the full contents of a virtual text file by its exact path,
    raising an error if no file exists at that path."""
    if path not in _files:
        raise FileNotFoundError(path)
    return _files[path]


@mcp.tool()
def files_write_file(path: str, content: str) -> str:
    """Write (create or overwrite) a virtual text file at the given path.
    Any existing content at that path is replaced entirely rather than
    appended to, and intermediate directories in the path do not need to
    exist beforehand since this is a flat virtual filesystem."""
    _files[path] = content
    return f"wrote {len(content)} bytes to {path}"


@mcp.tool()
def files_list_dir(prefix: str = "/") -> list[str]:
    """List virtual file paths starting with the given prefix."""
    return [p for p in _files if p.startswith(prefix)]


# ---------------------------------------------------------------------------
# Misc tools
# ---------------------------------------------------------------------------


@mcp.tool()
def weather_get_forecast(city: str) -> dict:
    """Return a (mocked) weather forecast for a city."""
    return {"city": city, "condition": "partly cloudy", "high_c": 24, "low_c": 15}


@mcp.tool()
def math_evaluate(expression: str) -> float:
    """Evaluate a simple arithmetic expression, e.g. '2 * (3 + 4)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("expression contains disallowed characters")
    return eval(expression, {"__builtins__": {}}, {})


# ---------------------------------------------------------------------------
# Inventory tools
# ---------------------------------------------------------------------------


@mcp.tool()
def inventory_check_stock(sku: str, warehouse: str = "", include_reserved: bool = False) -> dict:
    """Check the on-hand quantity for a product SKU, optionally scoped to a
    specific warehouse. Returns the current quantity, warehouse location,
    and whether the SKU is known to the inventory system at all. When
    include_reserved is true, the returned quantity also accounts for units
    that have already been reserved against open orders but not yet shipped."""
    item = _inventory.get(sku)
    if item is None:
        return {"sku": sku, "found": False, "quantity": 0}
    if warehouse and item.get("warehouse") != warehouse:
        return {"sku": sku, "found": False, "quantity": 0, "warehouse": warehouse}
    return {**item, "found": True, "include_reserved": include_reserved}


@mcp.tool()
def inventory_reserve_stock(sku: str, quantity: int, order_id: str, hold_expires_minutes: int = 30) -> dict:
    """Reserve a quantity of a SKU against a specific order id, decrementing
    on-hand inventory. The reservation automatically expires and releases
    the stock back to the pool after hold_expires_minutes if the order isn't
    confirmed. Fails if there isn't enough stock available to cover the
    requested quantity."""
    item = _inventory.setdefault(sku, {"sku": sku, "quantity": 0, "warehouse": "east-1"})
    if item["quantity"] < quantity:
        raise ValueError(f"insufficient stock for {sku}: have {item['quantity']}, need {quantity}")
    item["quantity"] -= quantity
    return {
        "sku": sku,
        "order_id": order_id,
        "reserved": quantity,
        "remaining": item["quantity"],
        "hold_expires_minutes": hold_expires_minutes,
    }


@mcp.tool()
def inventory_restock(sku: str, quantity: int, supplier: str = "", unit_cost: float = 0.0) -> dict:
    """Increase on-hand quantity for a SKU after a restock/purchase order is
    received, optionally recording which supplier fulfilled it and the
    per-unit cost paid, which feeds into landed-cost and margin reporting."""
    item = _inventory.setdefault(sku, {"sku": sku, "quantity": 0, "warehouse": "east-1"})
    item["quantity"] += quantity
    item["last_supplier"] = supplier
    item["last_unit_cost"] = unit_cost
    return item


# ---------------------------------------------------------------------------
# Billing tools
# ---------------------------------------------------------------------------


@mcp.tool()
def billing_create_invoice(
    customer_id: str, line_items: list[dict], currency: str = "USD", due_date: str = "", notes: str = ""
) -> dict:
    """Create a new invoice for a customer from a list of line items (each a
    dict with description/quantity/unit_price), returning the generated
    invoice id and computed total. An optional due date (YYYY-MM-DD) and
    free-form notes are stored on the invoice and surfaced on the customer
    -facing PDF, but do not affect how the total is calculated."""
    invoice_id = f"INV-{len(_invoices) + 1}"
    total = sum(li.get("quantity", 1) * li.get("unit_price", 0) for li in line_items)
    _invoices[invoice_id] = {
        "id": invoice_id,
        "customer_id": customer_id,
        "line_items": line_items,
        "currency": currency,
        "total": total,
        "amount_paid": 0,
        "status": "unpaid",
        "due_date": due_date,
        "notes": notes,
    }
    return _invoices[invoice_id]


@mcp.tool()
def billing_apply_payment(
    invoice_id: str,
    amount: float,
    payment_method: Literal["card", "bank_transfer", "paypal"] = "card",
    reference: str = "",
) -> dict:
    """Apply a payment of a given amount and method to an existing invoice,
    updating its paid status once the full total has been covered. An
    optional external reference (e.g. a payment processor transaction id)
    can be attached for reconciliation against the payment gateway's records."""
    if invoice_id not in _invoices:
        raise ValueError(f"no such invoice: {invoice_id}")
    invoice = _invoices[invoice_id]
    invoice["amount_paid"] += amount
    invoice["payment_method"] = payment_method
    invoice["payment_reference"] = reference
    invoice["status"] = "paid" if invoice["amount_paid"] >= invoice["total"] else "partial"
    return invoice


@mcp.tool()
def billing_issue_refund(
    invoice_id: str, amount: float, reason: str = "", restock_items: bool = False
) -> dict:
    """Issue a partial or full refund against a previously paid invoice,
    recording the reason for audit purposes. When restock_items is true, the
    quantities on the invoice's line items are also returned to inventory,
    which is appropriate for a returned-goods refund but not for a
    goodwill/billing-error refund."""
    if invoice_id not in _invoices:
        raise ValueError(f"no such invoice: {invoice_id}")
    invoice = _invoices[invoice_id]
    invoice["amount_paid"] -= amount
    invoice.setdefault("refunds", []).append({"amount": amount, "reason": reason, "restock_items": restock_items})
    return invoice


@mcp.tool()
def billing_get_invoice(invoice_id: str, include_line_items: bool = True) -> dict:
    """Fetch the full current state of an invoice by id, including its
    payment status and any refunds issued against it. Set
    include_line_items to false to omit the (potentially large) line-items
    array when you only need summary fields like status and total."""
    invoice = _invoices.get(invoice_id)
    if invoice is None:
        raise ValueError(f"no such invoice: {invoice_id}")
    if not include_line_items:
        return {k: v for k, v in invoice.items() if k != "line_items"}
    return invoice


# ---------------------------------------------------------------------------
# HR tools
# ---------------------------------------------------------------------------


@mcp.tool()
def hr_create_employee(
    name: str, email: str, department: str, title: str = "", manager_id: str = "", start_date: str = ""
) -> dict:
    """Onboard a new employee record with name, email, department, and
    optional job title. An optional manager_id links the new hire into the
    reporting hierarchy, and an optional start_date (YYYY-MM-DD) is used to
    schedule onboarding tasks and benefits enrollment deadlines."""
    employee_id = str(uuid.uuid4())[:8]
    _employees[employee_id] = {
        "id": employee_id,
        "name": name,
        "email": email,
        "department": department,
        "title": title,
        "manager_id": manager_id,
        "start_date": start_date,
        "status": "active",
    }
    return _employees[employee_id]


@mcp.tool()
def hr_update_employee_status(
    employee_id: str, status: Literal["active", "on_leave", "terminated"], effective_date: str = ""
) -> dict:
    """Update an employee's employment status, e.g. when they go on leave,
    return from leave, or are offboarded. An optional effective_date
    (YYYY-MM-DD) records when the status change actually takes effect,
    which may be in the future for a scheduled termination or leave start."""
    if employee_id not in _employees:
        raise ValueError(f"no such employee: {employee_id}")
    _employees[employee_id]["status"] = status
    _employees[employee_id]["status_effective_date"] = effective_date
    return _employees[employee_id]


@mcp.tool()
def hr_request_time_off(
    employee_id: str,
    start_date: str,
    end_date: str,
    reason: str = "",
    leave_type: Literal["vacation", "sick", "unpaid", "other"] = "vacation",
) -> dict:
    """File a time-off request for an employee covering a start and end date
    (YYYY-MM-DD), with an optional reason and a leave_type that determines
    which balance the days are deducted from and whether manager approval
    is required before the request is confirmed."""
    request_id = f"PTO-{len(_time_off_requests) + 1}"
    _time_off_requests[request_id] = {
        "id": request_id,
        "employee_id": employee_id,
        "start_date": start_date,
        "end_date": end_date,
        "reason": reason,
        "leave_type": leave_type,
        "status": "pending",
    }
    return _time_off_requests[request_id]


# ---------------------------------------------------------------------------
# Analytics tools
# ---------------------------------------------------------------------------


@mcp.tool()
def analytics_run_query(dataset: str, sql: str, limit: int = 100, timeout_seconds: int = 30) -> dict:
    """Run a read-only SQL query against a named analytics dataset and
    return a (mocked) row count and sample of the query that was executed,
    capped at the given row limit. The query is killed and an error
    returned if it runs longer than timeout_seconds, to protect the
    underlying warehouse from runaway scans."""
    return {"dataset": dataset, "sql": sql, "limit": limit, "timeout_seconds": timeout_seconds, "row_count": 0, "rows": []}


@mcp.tool()
def analytics_get_dashboard(dashboard_id: str, include_widget_data: bool = False) -> dict:
    """Fetch metadata and widget layout for a saved analytics dashboard by
    id, including its title and the metrics it tracks. Set
    include_widget_data to true to also fetch each widget's current data
    (slower, since it re-runs every underlying query)."""
    return {"id": dashboard_id, "title": "Untitled dashboard", "widgets": [], "include_widget_data": include_widget_data}


@mcp.tool()
def analytics_create_report(
    name: str,
    metrics: list[str],
    date_range: str = "last_7_days",
    schedule: Literal["once", "daily", "weekly", "monthly"] = "once",
) -> dict:
    """Create a report over a list of named metrics for a given date range
    (e.g. 'last_7_days', 'last_30_days', 'month_to_date'). When schedule is
    anything other than 'once', the report is regenerated and re-delivered
    automatically on that cadence rather than being a one-time snapshot."""
    report_id = f"REPORT-{len(_reports) + 1}"
    _reports[report_id] = {
        "id": report_id,
        "name": name,
        "metrics": metrics,
        "date_range": date_range,
        "schedule": schedule,
    }
    return _reports[report_id]


# ---------------------------------------------------------------------------
# Notification tools
# ---------------------------------------------------------------------------


@mcp.tool()
def notifications_send_email(
    to: str, subject: str, body: str, cc: list[str] | None = None, is_html: bool = False
) -> dict:
    """Send a transactional email to an address, with an optional list of
    cc recipients, returning a generated message id. Set is_html to true
    when body contains HTML markup rather than plain text, so the mail
    provider renders it correctly instead of escaping the tags."""
    return {"message_id": str(uuid.uuid4())[:8], "to": to, "subject": subject, "cc": cc or [], "is_html": is_html}


@mcp.tool()
def notifications_send_sms(to: str, message: str, sender_id: str = "") -> dict:
    """Send a short SMS text message to a phone number, returning a
    generated message id. An optional sender_id overrides the default
    alphanumeric sender name shown to the recipient, where the destination
    carrier supports it."""
    return {"message_id": str(uuid.uuid4())[:8], "to": to, "message": message, "sender_id": sender_id}


@mcp.tool()
def notifications_send_push(
    user_id: str, title: str, body: str, priority: Literal["normal", "high"] = "normal"
) -> dict:
    """Send a mobile push notification with a title and body to a specific
    user id's registered devices. High-priority pushes bypass the
    recipient's notification batching/quiet-hours settings and should be
    reserved for time-sensitive alerts."""
    return {
        "message_id": str(uuid.uuid4())[:8],
        "user_id": user_id,
        "title": title,
        "body": body,
        "priority": priority,
    }


# ---------------------------------------------------------------------------
# Calendar tools
# ---------------------------------------------------------------------------


@mcp.tool()
def calendar_create_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list[str] | None = None,
    location: str = "",
    recurrence: str = "",
) -> dict:
    """Create a calendar event with a title, ISO 8601 start/end time, an
    optional list of attendee emails, and an optional location. An optional
    recurrence rule (an RFC 5545 RRULE string, e.g. 'FREQ=WEEKLY;COUNT=10')
    turns this into a repeating series rather than a single one-off event."""
    event_id = str(uuid.uuid4())[:8]
    _events[event_id] = {
        "id": event_id,
        "title": title,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees or [],
        "location": location,
        "recurrence": recurrence,
    }
    return _events[event_id]


@mcp.tool()
def calendar_find_free_slots(
    attendees: list[str], duration_minutes: int = 30, date: str = "", working_hours_only: bool = True
) -> list[dict]:
    """Find (mocked) free meeting slots of a given duration for a list of
    attendee emails on a specific date (YYYY-MM-DD, defaults to today).
    When working_hours_only is true, slots outside each attendee's
    configured working hours (as set in their calendar preferences) are
    excluded even if their calendar shows them as free."""
    return [
        {
            "start_time": f"{date or 'today'}T09:00:00",
            "end_time": f"{date or 'today'}T09:{duration_minutes:02d}:00",
            "working_hours_only": working_hours_only,
        }
    ]


# ---------------------------------------------------------------------------
# Project management tools
# ---------------------------------------------------------------------------


@mcp.tool()
def project_create_task(
    project_id: str,
    title: str,
    assignee: str = "",
    priority: Literal["low", "medium", "high"] = "medium",
    due_date: str = "",
    labels: list[str] | None = None,
) -> dict:
    """Create a new task under a project with an optional assignee,
    priority, and due date (YYYY-MM-DD). Optional labels (free-form tags
    like 'bug' or 'needs-design') are used for board filtering and don't
    affect the task's workflow status or notifications."""
    task_id = f"TASK-{len(_tasks) + 1}"
    _tasks[task_id] = {
        "id": task_id,
        "project_id": project_id,
        "title": title,
        "assignee": assignee,
        "priority": priority,
        "due_date": due_date,
        "labels": labels or [],
        "status": "todo",
    }
    return _tasks[task_id]


@mcp.tool()
def project_update_task_status(
    task_id: str, status: Literal["todo", "in_progress", "done"], comment: str = ""
) -> dict:
    """Update the workflow status of a project task by id. An optional
    comment is appended to the task's activity log alongside the status
    change, visible to everyone with access to the project."""
    if task_id not in _tasks:
        raise ValueError(f"no such task: {task_id}")
    _tasks[task_id]["status"] = status
    _tasks[task_id].setdefault("activity_log", []).append({"status": status, "comment": comment})
    return _tasks[task_id]


@mcp._mcp_server.list_tools()
async def _list_tools_with_cache_hints() -> types.ListToolsResult:
    """Override FastMCP's default ListTools handler to attach ttlMs/cacheScope
    cache hints (2026-07-28 spec) via `_meta`, so clients can cache the
    catalog host-side instead of re-issuing tools/list every turn."""
    tools = await mcp.list_tools()
    return types.ListToolsResult(
        tools=tools,
        _meta={"ttlMs": TOOLS_LIST_TTL_MS, "cacheScope": TOOLS_LIST_CACHE_SCOPE},
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
