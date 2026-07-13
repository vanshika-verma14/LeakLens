"""Generate + validate the labelled ground-truth corpus (corpus/corpus.jsonl).

Each row is ``{"id", "text", "type", "key_entities"}``. ``key_entities`` are verbatim
substrings of ``text`` (the validator asserts this), so the recovery metric can score
them without the corpus ever committing to a normalization rule (that lives in
metrics.py). Every secret / PII value is a reserved, non-functional fake — RFC-5737 IPs
(192.0.2.x), ``@example.com`` emails, ``555-01xx`` phones, ``…EXAMPLE`` AWS keys,
``sk-test-…`` tokens, fictional names/addresses.

Design constraints enforced here (see the curation feedback in the plan):
  * **Frame cap** — every category is assembled from many distinct sentence *frames*;
    a per-frame cap (default 3) means no frame appears more than ~3 times in the corpus.
  * **Unique credential values** — every password / key / token / connection-string
    secret is generated once and never reused, so the credential category measures
    recovery honestly rather than re-scoring the same string.
  * **Content-bearing entities only** — no generic sentence-initial framing words.

Generation is deterministic under ``--seed`` (reproducibility, NFR-3). The output is a
*candidate* corpus meant to be human-curated, not a finished artifact.

Usage:
    python corpus/build_corpus.py                 # build -> corpus/corpus.jsonl, validate
    python corpus/build_corpus.py --dry-run       # build + validate + report, write nothing
    python corpus/build_corpus.py --seed 7        # different deterministic draw
    python corpus/build_corpus.py --validate F    # validate an existing jsonl, no build
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

CATEGORIES = ("plain", "pii", "credential", "structured")
PREFIX = {"plain": "plain", "pii": "pii", "credential": "cred", "structured": "struct"}
TARGET_PER_CAT = 60
FRAME_CAP = 3  # no sentence frame appears more than this many times per category

# ---------------------------------------------------------------------------
# Reserved / non-functional fake value pools (see fake-data rule in the plan).
# ---------------------------------------------------------------------------
NAMES = [
    "Priya Sharma", "Daniel Reyes", "Aisha Khan", "Marcus Bell", "Lena Fischer",
    "Omar Haddad", "Sofia Rossi", "Kenji Tanaka", "Grace Okoro", "Ivan Petrov",
    "Nadia Ali", "Tom Becker", "Elena Vasquez", "Raj Malhotra", "Chloe Dubois",
    "Samuel Adeyemi", "Mei Lin", "Farah Nasser", "Diego Morales", "Hannah Cohen",
]
PHONES = [f"555-01{n:02d}" for n in range(10, 30)]  # 555-0110 .. 555-0129 (reserved)
STREETS = [
    "12 Example Ave", "48 Sample Street", "7 Placeholder Road", "155 Test Lane",
    "1600 Example Ave", "23 Fictional Way", "89 Dummy Court", "301 Mock Boulevard",
    "5 Demo Terrace", "64 Sandbox Drive",
]
CITIES = ["Springfield", "Fairview", "Riverton", "Lakeside", "Greenville",
          "Ashford", "Brookline", "Westborough", "Millbrook", "Oakdale"]
ZIPS = [f"{n:05d}" for n in (100, 220, 3310, 4405, 55010, 6120, 7008, 8890, 9005, 12000)]
DOBS = ["1988-02-17", "1990-04-12", "1979-11-03", "1995-07-28", "1983-01-09",
        "2001-09-30", "1992-12-15", "1975-06-21", "1998-03-06", "1986-10-24"]
THINGS = ["outstanding invoice", "delivery window", "refund request",
          "account update", "support ticket", "renewal notice", "onboarding call"]
ENVS = ["staging", "production", "sandbox", "development"]

PLAIN_SUBJECTS = [
    "analytics pipeline", "mobile app", "onboarding flow", "search index",
    "billing service", "recommendation engine", "data warehouse", "CI pipeline",
    "design system", "payment gateway", "notification service", "reporting dashboard",
    "authentication layer", "content management system", "inventory tracker",
]
PCTS = ["12 percent", "18 percent", "9 percent", "27 percent", "33 percent", "6 percent"]

# Standalone everyday sentences (each is its own one-off frame). Entities are the
# specific, content-bearing spans only — no generic sentence-initial framing words.
PLAIN_STATIC = [
    ("The museum's new exhibit on deep-sea life opens to the public next Tuesday.",
     ["deep-sea life", "exhibit"]),
    ("Volunteers planted over two hundred oak saplings along the river trail.",
     ["oak saplings", "river trail"]),
    ("The recipe calls for slow-roasting the vegetables before adding the broth.",
     ["slow-roasting", "vegetables", "broth"]),
    ("Heavy rain is expected across the northern hills through the weekend.",
     ["northern hills"]),
    ("The orchestra rehearsed the symphony's final movement late into the evening.",
     ["orchestra", "symphony", "final movement"]),
    ("Local farmers reported a strong harvest despite the unusually dry summer.",
     ["harvest", "dry summer"]),
    ("The library extended its opening hours during the exam season.",
     ["library", "opening hours", "exam season"]),
    ("Researchers observed the migrating cranes returning to the wetlands early.",
     ["migrating cranes", "wetlands"]),
    ("The bakery on the corner sells out of sourdough by mid-morning.",
     ["bakery", "sourdough"]),
    ("The documentary traces the history of the coastal railway line.",
     ["documentary", "coastal railway line"]),
    ("The science fair drew crowds to the school gymnasium on Saturday.",
     ["science fair", "school gymnasium"]),
    ("The hiking club mapped a new route through the pine forest ridge.",
     ["hiking club", "pine forest ridge"]),
    ("A sudden cold snap delayed the spring planting by nearly two weeks.",
     ["cold snap", "spring planting"]),
    ("The film festival announced its lineup of documentaries and short features.",
     ["film festival", "documentaries", "short features"]),
    ("The annual ride across the valley drew hundreds of cyclists at dawn.",
     ["annual ride", "valley", "cyclists"]),
    ("The chef swapped butter for olive oil to lighten the classic sauce.",
     ["chef", "olive oil", "classic sauce"]),
    ("An old lighthouse on the cape was restored by local craftspeople.",
     ["lighthouse", "cape", "craftspeople"]),
    ("The debate team argued both sides of the housing policy question.",
     ["debate team", "housing policy"]),
    ("Snowmelt from the mountains feeds the reservoir through early summer.",
     ["Snowmelt", "reservoir"]),
    ("The gallery's photography show drew visitors from neighboring towns.",
     ["gallery", "photography show"]),
]

# Subject-based frames. ``{s}`` is a PLAIN_SUBJECT; a few use ``{art} {pct}``. The
# shapes vary deliberately (passive, fronted adverbial, question, subordinate-clause
# lead, appositive, gerund subject) so no frame dominates a draw.
PLAIN_TEMPLATES = [
    ("The team migrated the {s} from nightly batch jobs to streaming last spring.",
     ["{s}", "batch jobs", "streaming"]),
    ("Engineering cut {s} latency after a caching rewrite this quarter.",
     ["{s}", "latency", "caching"]),
    ("Across three regions, the {s} now serves traffic without downtime.",
     ["{s}", "three regions"]),
    ("Documentation for the {s} was refreshed during the last sprint.",
     ["{s}", "sprint"]),
    ("The quarterly review noted {art} {pct} improvement in {s} reliability.",
     ["{s}", "{pct}", "reliability"]),
    ("Why did the {s} slow down? A missing index turned out to be the cause.",
     ["{s}", "missing index"]),
    ("Once the {s} moved to read replicas, tail latency dropped sharply.",
     ["{s}", "read replicas", "tail latency"]),
    ("Our on-call team rewrote the {s} retry logic to stop cascading failures.",
     ["{s}", "retry logic", "cascading failures"]),
    ("Adoption of the {s} climbed steadily after the redesign shipped.",
     ["{s}", "redesign"]),
    ("The {s}, long a source of pager alerts, finally stabilized this month.",
     ["{s}", "pager alerts"]),
    ("Product and design reviewed the {s} roadmap before the offsite.",
     ["{s}", "roadmap", "offsite"]),
    ("Migrating the {s} off the legacy queue freed up two engineers.",
     ["{s}", "legacy queue"]),
    ("By caching hot keys, the {s} shaved milliseconds off every request.",
     ["{s}", "hot keys", "milliseconds"]),
    ("The {s} passed its load test at ten thousand concurrent users.",
     ["{s}", "load test", "ten thousand concurrent users"]),
    ("The {s} team shipped a redesign that cut support tickets in half.",
     ["{s}", "redesign", "support tickets"]),
    ("After the outage, the {s} added circuit breakers to fail fast.",
     ["{s}", "outage", "circuit breakers"]),
    ("Nobody expected the {s} to become the most-used feature this year.",
     ["{s}", "most-used feature"]),
    ("The {s} finally got proper dashboards and alerting last week.",
     ["{s}", "dashboards", "alerting"]),
    ("We deprecated the old {s} client in favor of the new SDK.",
     ["{s}", "SDK"]),
    ("Load on the {s} doubled during the holiday sale without incident.",
     ["{s}", "holiday sale"]),
    ("The {s} now backs up to cold storage every night.",
     ["{s}", "cold storage"]),
    ("A flaky test in the {s} suite was quarantined until the fix landed.",
     ["{s}", "flaky test"]),
]


def _indef_article(phrase: str) -> str:
    """Return 'a' or 'an' for the phrase that follows, handling leading numbers.

    Number rule covers 1-2 digit values (all we place before an article): the
    spoken form starts with a vowel sound for 8 (eight), 11 (eleven), 18
    (eighteen), and 80-89 (eighty-...). Everything else defaults to 'a'.
    """
    first = phrase.strip().split()[0]
    if first[:1].isdigit():
        n = re.match(r"\d+", first).group(0)
        if n in ("8", "11", "18") or (len(n) == 2 and n[0] == "8"):
            return "an"
        return "a"
    return "an" if first[:1].lower() in "aeiou" else "a"


# ---------------------------------------------------------------------------
# Unique fake-secret generators (credential category). Each value is minted once.
# ---------------------------------------------------------------------------
_PW_WORDS = ["Vault", "Cloud", "North", "Pine", "Ember", "Quartz", "Delta", "Raven",
             "Onyx", "Cobalt", "Zephyr", "Falcon", "Basil", "Nimbus", "Cedar", "Slate",
             "Coral", "Vertex", "Lunar", "Harbor", "Aspen", "Cove", "Flint", "Grove",
             "Marsh", "Ridge", "Sable", "Tundra", "Willow", "Yukon"]
_PW_SYMS = "!#$%^&*-_+"          # deliberately excludes '@' so URI passwords stay valid
_UPPER36 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_HEX = "0123456789abcdef"


def _mk_password(rng: random.Random) -> str:
    return (f"{rng.choice(_PW_WORDS)}{rng.randint(10, 99)}"
            f"{rng.choice(_PW_SYMS)}{rng.choice(_PW_WORDS).lower()[:4]}")


def _mk_aws_key(rng: random.Random) -> str:
    body = "".join(rng.choice(_UPPER36) for _ in range(9))
    return f"AKIA{body}EXAMPLE"


def _mk_token(rng: random.Random) -> str:
    body = "".join(rng.choice(_ALNUM) for _ in range(24))
    return f"sk-test-{body}"


def _mk_ssh_key(rng: random.Random) -> str:
    body = "".join(rng.choice(_HEX) for _ in range(6))
    return f"ssh-ed25519-EXAMPLE-{body}"


def _unique(gen, used: set) -> str:
    """Draw from ``gen`` until a not-yet-seen value appears; record and return it."""
    for _ in range(10_000):
        v = gen()
        if v not in used:
            used.add(v)
            return v
    raise SystemExit("could not mint a unique secret value")


# ---------------------------------------------------------------------------
# Frame builders. Each returns a list of (frame_id, [(text, entities), ...]).
# ---------------------------------------------------------------------------
def _plain(rng: random.Random):
    groups = []
    for idx, (text, ents) in enumerate(PLAIN_STATIC):
        groups.append((f"plain-static-{idx:02d}", [(text, list(ents))]))
    for idx, (tpl, ent_tpls) in enumerate(PLAIN_TEMPLATES):
        insts = []
        for s in PLAIN_SUBJECTS:
            slots = {"s": s}
            if "{pct}" in tpl:
                pct = rng.choice(PCTS)
                slots["pct"] = pct
                slots["art"] = _indef_article(pct)
            text = tpl.format(**slots)
            ents = [e.format(**slots) for e in ent_tpls]
            insts.append((text, ents))
        groups.append((f"plain-tpl-{idx:02d}", insts))
    return groups


def _pii(rng: random.Random):
    groups = []
    email_frames = [
        "Email {name} at {email} about the {thing}.",
        "Send the receipt to {name} at {email} by end of day.",
        "{name}'s contact address on file is {email}.",
        "Loop in {name} at {email} on the thread.",
        "Forward the report to {email}, addressed to {name}.",
        "Update the mailing preferences for {name} at {email}.",
    ]
    for i, tpl in enumerate(email_frames):
        insts = []
        for name in NAMES:
            email = name.lower().replace(" ", ".") + "@example.com"
            text = tpl.format(name=name, email=email, thing=rng.choice(THINGS))
            insts.append((text, [name, email]))
        groups.append((f"pii-email-{i}", insts))

    phone_frames = [
        "Call {name} on {phone} to confirm the {thing}.",
        "{name} can be reached at {phone} after noon.",
        "Text {name} at {phone} once the parcel ships.",
        "The callback number for {name} is {phone}.",
        "Reach {name} on {phone} regarding the {thing}.",
        "Leave a voicemail for {name} at {phone}.",
    ]
    for i, tpl in enumerate(phone_frames):
        insts = []
        for j, name in enumerate(NAMES):
            phone = PHONES[j % len(PHONES)]
            text = tpl.format(name=name, phone=phone, thing=rng.choice(THINGS))
            insts.append((text, [name, phone]))
        groups.append((f"pii-phone-{i}", insts))

    addr_frames = [
        "Ship the parcel to {name}, {street}, {city} {zip}.",
        "{name} moved to {street}, {city} {zip} last month.",
        "The billing address for {name} is {street}, {city} {zip}.",
        "Deliver the documents to {name} at {street}, {city} {zip}.",
        "Register {name} at {street}, {city} {zip} for pickup.",
    ]
    for i, tpl in enumerate(addr_frames):
        insts = []
        for j, name in enumerate(NAMES):
            street, city, z = STREETS[j % len(STREETS)], CITIES[j % len(CITIES)], ZIPS[j % len(ZIPS)]
            text = tpl.format(name=name, street=street, city=city, zip=z)
            insts.append((text, [name, street, city, z]))
        groups.append((f"pii-addr-{i}", insts))

    dob_frames = [
        "The account for {name} lists a date of birth of {dob}.",
        "{name} was born on {dob} according to the record.",
        "Verify {name}'s identity using the birth date {dob}.",
        "On file: {name}, date of birth {dob}.",
        "{name}'s date of birth is recorded as {dob}.",
    ]
    for i, tpl in enumerate(dob_frames):
        insts = []
        for j, name in enumerate(NAMES):
            dob = DOBS[j % len(DOBS)]
            insts.append((tpl.format(name=name, dob=dob), [name, dob]))
        groups.append((f"pii-dob-{i}", insts))
    return groups


def _credential(rng: random.Random):
    used: set = set()
    groups = []
    accts = ["admin", "root", "database", "service", "staging", "backup", "deploy", "readonly"]
    when = ["Friday's audit", "the next deploy", "the security review", "end of week"]

    pw_frames = [
        "Reset the {acct} password to {pw} before {when}.",
        "The {acct} password is {pw}; store it in the vault.",
        "Set the {acct} password to {pw} on first login.",
        "Rotate the {acct} password to {pw} this cycle.",
        "Temporary {acct} password: {pw} (change after use).",
        "Use {pw} as the {acct} password for the staging box.",
    ]
    for i, tpl in enumerate(pw_frames):
        insts = []
        for _ in range(FRAME_CAP):
            acct = rng.choice(accts)
            pw = _unique(lambda: _mk_password(rng), used)
            text = tpl.format(acct=acct, pw=pw, when=rng.choice(when))
            insts.append((text, [f"{acct} password", pw]))
        groups.append((f"cred-pw-{i}", insts))

    aws_frames = [
        "Rotate the AWS access key {key} in the {env} account.",
        "The AWS access key {key} was revoked after the review.",
        "Provision a new AWS access key {key} for the {env} role.",
        "Do not commit the AWS access key {key} to the repository.",
    ]
    for i, tpl in enumerate(aws_frames):
        insts = []
        for _ in range(FRAME_CAP):
            key = _unique(lambda: _mk_aws_key(rng), used)
            insts.append((tpl.format(key=key, env=rng.choice(ENVS)), ["AWS access key", key]))
        groups.append((f"cred-aws-{i}", insts))

    token_frames = [
        "The {env} API token is {tok} and must not be shared.",
        "Store the API token {tok} in the secrets manager.",
        "Regenerate the API token {tok} after the incident.",
        "The webhook uses the API token {tok} for authentication.",
    ]
    for i, tpl in enumerate(token_frames):
        insts = []
        for _ in range(FRAME_CAP):
            tok = _unique(lambda: _mk_token(rng), used)
            insts.append((tpl.format(tok=tok, env=rng.choice(ENVS)), ["API token", tok]))
        groups.append((f"cred-tok-{i}", insts))

    ssh_frames = [
        "Store the deploy key {key} in the vault, not in source control.",
        "Add the deploy key {key} to the {env} server.",
        "The deploy key {key} grants access to the release pipeline.",
        "Revoke the deploy key {key} once the contractor leaves.",
    ]
    for i, tpl in enumerate(ssh_frames):
        insts = []
        for _ in range(FRAME_CAP):
            key = _unique(lambda: _mk_ssh_key(rng), used)
            insts.append((tpl.format(key=key, env=rng.choice(ENVS)), ["deploy key", key]))
        groups.append((f"cred-ssh-{i}", insts))

    conn_frames = [
        "The connection string is postgres://svc:{pw}@db.example.com:5432/app.",
        "Connect using the connection string mysql://app:{pw}@db.example.com:3306/main.",
        "The connection string redis://cache:{pw}@db.example.com:6379/0 is in the config.",
    ]
    for i, tpl in enumerate(conn_frames):
        insts = []
        for _ in range(FRAME_CAP):
            pw = _unique(lambda: _mk_password(rng), used)  # no '@' by construction
            insts.append((tpl.format(pw=pw), ["connection string", pw]))
        groups.append((f"cred-conn-{i}", insts))
    return groups


def _structured(rng: random.Random):
    oids = [f"ORD-2026-{n:05d}" for n in (817, 1204, 3391, 4420, 5507, 6612, 7003, 8890,
                                          9015, 10120, 11233, 12044, 13567, 14890, 15011)]
    invs = [f"INV-{n}" for n in (4471, 4482, 4519, 4603, 4712, 4820, 4931, 5044, 5188,
                                 5210, 5377, 5499, 5602, 5715, 5888)]
    accts = [f"{n:012d}" for n in (123456789, 987654321, 456123789, 741852963, 369258147,
                                   852456123, 159357486, 753951456, 246813579, 135792468,
                                   864209753, 975318642, 112233445, 556677889, 998877665)]
    dates = ["2026-03-14", "2025-11-02", "2026-01-31", "2025-08-19", "2026-05-06",
             "2025-12-25", "2026-02-08", "2025-10-11", "2026-04-23", "2025-09-30",
             "2026-06-17", "2025-07-04", "2026-07-01", "2025-06-13", "2026-08-28"]
    amts = ["$1,249.50", "$980.00", "$3,410.75", "$56.20", "$12,000.00", "$742.99",
            "$205.10", "$8,675.30", "$49.95", "$1,000.00", "$333.33", "$27,540.60",
            "$610.00", "$4,299.99", "$88.40"]
    hosts = ["db-prod", "cache-01", "api-gw", "queue-02", "index-03", "auth-01"]
    ips = [f"192.0.2.{n}" for n in (10, 23, 47, 88, 101, 134, 150, 172, 199, 210,
                                    5, 33, 66, 120, 240)]
    ports = ["5432", "6379", "8080", "443", "27017", "9092", "3306", "1521"]
    cfg = [("MAX_RETRIES", "5"), ("CACHE_TTL", "3600"), ("BATCH_SIZE", "128"),
           ("TIMEOUT_MS", "2000"), ("POOL_SIZE", "16"), ("LOG_LEVEL", "info"),
           ("RATE_LIMIT", "100"), ("WORKER_COUNT", "8"), ("QUEUE_DEPTH", "512"),
           ("SHARD_COUNT", "4"), ("RETRY_BACKOFF", "250"), ("MAX_CONN", "64"),
           ("FLUSH_EVERY", "1000"), ("TTL_DAYS", "30"), ("PAGE_SIZE", "50")]
    txns = [f"TXN-{n}" for n in (88213, 88301, 88477, 88590, 88612, 88734, 88801, 88950,
                                 89012, 89133, 89244, 89377, 89401, 89588, 89610)]
    n = 8  # instances per frame before the cap trims to FRAME_CAP
    groups = []

    def add(fid, insts):
        groups.append((fid, insts))

    # ---- orders ----
    add("struct-ord-0", [(f"Order {oids[i]} shipped on {dates[i]} for a total of {amts[(i * 2) % 15]}.",
                          ["Order", oids[i], dates[i], amts[(i * 2) % 15]]) for i in range(n)])
    add("struct-ord-1", [(f"Order {oids[(i + 4) % 15]} was refunded {amts[i]} on {dates[(i + 2) % 15]}.",
                          ["Order", oids[(i + 4) % 15], amts[i], dates[(i + 2) % 15]]) for i in range(n)])
    add("struct-ord-2", [(f"We fulfilled order {oids[(i + 7) % 15]} on {dates[(i + 5) % 15]}.",
                          ["order", oids[(i + 7) % 15], dates[(i + 5) % 15]]) for i in range(n)])
    add("struct-ord-3", [(f"The invoice for order {oids[(i + 2) % 15]} totals {amts[(i + 6) % 15]}.",
                          ["order", oids[(i + 2) % 15], amts[(i + 6) % 15]]) for i in range(n)])
    add("struct-ord-4", [(f"Order {oids[(i + 9) % 15]} was cancelled before shipping on {dates[(i + 8) % 15]}.",
                          ["Order", oids[(i + 9) % 15], dates[(i + 8) % 15]]) for i in range(n)])

    # ---- invoices / accounts ----
    add("struct-inv-0", [(f"Invoice {invs[i]} lists account number {accts[i]} due on {dates[(i + 3) % 15]}.",
                          ["Invoice", invs[i], "account number", accts[i], dates[(i + 3) % 15]]) for i in range(n)])
    add("struct-inv-1", [(f"Invoice {invs[(i + 5) % 15]} for {amts[i]} was issued on {dates[(i + 1) % 15]}.",
                          ["Invoice", invs[(i + 5) % 15], amts[i], dates[(i + 1) % 15]]) for i in range(n)])
    add("struct-inv-2", [(f"The order references account number {accts[(i + 6) % 15]} on invoice {invs[(i + 2) % 15]}.",
                          ["account number", accts[(i + 6) % 15], "invoice", invs[(i + 2) % 15]]) for i in range(n)])
    add("struct-inv-3", [(f"Payment for invoice {invs[(i + 8) % 15]} of {amts[(i + 4) % 15]} cleared on {dates[(i + 7) % 15]}.",
                          ["invoice", invs[(i + 8) % 15], amts[(i + 4) % 15], dates[(i + 7) % 15]]) for i in range(n)])
    add("struct-inv-4", [(f"The credit note for invoice {invs[(i + 3) % 15]} was {amts[(i + 9) % 15]}.",
                          ["invoice", invs[(i + 3) % 15], amts[(i + 9) % 15]]) for i in range(n)])

    # ---- db host / ip / port ----
    add("struct-db-0", [(f"The database host {hosts[i % 6]} is reachable at {ips[i]} on port {ports[i % 8]}.",
                         ["database host", ips[i], f"port {ports[i % 8]}"]) for i in range(n)])
    add("struct-db-1", [(f"Point the client at {ips[(i + 3) % 15]}:{ports[(i + 1) % 8]} for the {hosts[(i + 2) % 6]} service.",
                         [hosts[(i + 2) % 6], ips[(i + 3) % 15]]) for i in range(n)])
    add("struct-db-2", [(f"The {hosts[(i + 4) % 6]} node moved to {ips[(i + 6) % 15]} during the migration.",
                         [hosts[(i + 4) % 6], ips[(i + 6) % 15]]) for i in range(n)])
    add("struct-db-3", [(f"Open port {ports[(i + 2) % 8]} on {ips[(i + 9) % 15]} for the {hosts[(i + 1) % 6]} replica.",
                         [f"port {ports[(i + 2) % 8]}", ips[(i + 9) % 15], hosts[(i + 1) % 6]]) for i in range(n)])

    # ---- config key/value ----
    add("struct-cfg-0", [(f"Set the config key {cfg[i][0]}={cfg[i][1]} in the staging environment.",
                          ["config key", f"{cfg[i][0]}={cfg[i][1]}"]) for i in range(n)])
    add("struct-cfg-1", [(f"The service reads {cfg[(i + 5) % 15][0]}={cfg[(i + 5) % 15][1]} from the environment file.",
                          [f"{cfg[(i + 5) % 15][0]}={cfg[(i + 5) % 15][1]}", "environment file"]) for i in range(n)])
    add("struct-cfg-2", [(f"Override {cfg[(i + 8) % 15][0]}={cfg[(i + 8) % 15][1]} for the load test.",
                          [f"{cfg[(i + 8) % 15][0]}={cfg[(i + 8) % 15][1]}", "load test"]) for i in range(n)])
    add("struct-cfg-3", [(f"The default {cfg[(i + 2) % 15][0]}={cfg[(i + 2) % 15][1]} was raised last release.",
                          [f"{cfg[(i + 2) % 15][0]}={cfg[(i + 2) % 15][1]}"]) for i in range(n)])

    # ---- transactions ----
    add("struct-txn-0", [(f"Transaction {txns[i]} of {amts[(i + 5) % 15]} settled on {dates[(i + 7) % 15]}.",
                          ["Transaction", txns[i], amts[(i + 5) % 15], dates[(i + 7) % 15]]) for i in range(n)])
    add("struct-txn-1", [(f"Transaction {txns[(i + 4) % 15]} was reversed on {dates[(i + 2) % 15]}.",
                          ["Transaction", txns[(i + 4) % 15], dates[(i + 2) % 15]]) for i in range(n)])
    add("struct-txn-2", [(f"A chargeback hit transaction {txns[(i + 7) % 15]} for {amts[(i + 3) % 15]}.",
                          ["transaction", txns[(i + 7) % 15], amts[(i + 3) % 15]]) for i in range(n)])
    add("struct-txn-3", [(f"Transaction {txns[(i + 9) % 15]} cleared {amts[(i + 8) % 15]} to the merchant.",
                          ["Transaction", txns[(i + 9) % 15], amts[(i + 8) % 15]]) for i in range(n)])
    return groups


_GENERATORS = {"plain": _plain, "pii": _pii, "credential": _credential, "structured": _structured}


def _select(frame_groups, rng: random.Random, target: int = TARGET_PER_CAT, cap: int = FRAME_CAP):
    """Cap each frame at ``cap`` unique instances, then shuffle and take ``target``.

    Returns list of (text, entities, frame_id). Because every frame contributes at
    most ``cap`` rows, no frame can appear more than ``cap`` times in the result.
    """
    pool = []
    seen_text: set = set()
    for fid, insts in frame_groups:
        seen_in_frame: set = set()
        uniq = []
        for text, ents in insts:
            if text in seen_in_frame:
                continue
            seen_in_frame.add(text)
            uniq.append((text, ents))
        rng.shuffle(uniq)
        for text, ents in uniq[:cap]:
            if text in seen_text:
                continue
            seen_text.add(text)
            pool.append((text, ents, fid))
    rng.shuffle(pool)
    if len(pool) < target:
        raise SystemExit(f"only {len(pool)} candidate rows, need {target}")
    return pool[:target]


def _build_tagged(seed: int = 42):
    """Build the corpus, returning list of (row_dict, frame_id)."""
    rng = random.Random(seed)
    tagged = []
    for cat in CATEGORIES:
        selected = _select(_GENERATORS[cat](rng), rng)
        for i, (text, ents, fid) in enumerate(selected, 1):
            tagged.append(({"id": f"{PREFIX[cat]}-{i:03d}", "text": text,
                            "type": cat, "key_entities": ents}, fid))
    return tagged


def build(seed: int = 42) -> list[dict]:
    """Build the corpus deterministically (drops the internal frame ids)."""
    return [row for row, _ in _build_tagged(seed)]


# ---------------------------------------------------------------------------
# Validation + fake-data lint
# ---------------------------------------------------------------------------
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_IPV4 = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
_RFC5737 = re.compile(r"^(192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)\d{1,3}$")
_ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")


def validate(rows: list[dict]) -> list[str]:
    """Return a list of human-readable problems; empty list == corpus is clean."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    counts = {c: 0 for c in CATEGORIES}
    cred_values: list[str] = []
    for r in rows:
        rid = r.get("id", "<no-id>")
        for key in ("id", "text", "type", "key_entities"):
            if key not in r:
                errors.append(f"{rid}: missing key '{key}'")
        if r.get("type") not in CATEGORIES:
            errors.append(f"{rid}: invalid type {r.get('type')!r}")
        else:
            counts[r["type"]] += 1
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate id")
        seen_ids.add(rid)

        text = r.get("text", "")
        ents = r.get("key_entities", [])
        if not isinstance(ents, list) or not ents:
            errors.append(f"{rid}: key_entities must be a non-empty list")
        else:
            for e in ents:
                if e not in text:
                    errors.append(f"{rid}: key_entity {e!r} is not a substring of text")
            if r.get("type") == "credential" and len(ents) >= 2:
                cred_values.append(ents[1])

        for m in _EMAIL.finditer(text):
            domain = m.group(1).lower()
            if not domain.endswith(_ALLOWED_EMAIL_DOMAINS):
                errors.append(f"{rid}: email domain {domain!r} is not a reserved example domain")
        for m in _IPV4.finditer(text):
            if not _RFC5737.match(m.group(1)):
                errors.append(f"{rid}: IP {m.group(1)!r} is outside RFC-5737 test ranges")

    for c in CATEGORIES:
        if counts[c] != TARGET_PER_CAT:
            errors.append(f"category {c}: has {counts[c]} rows, expected {TARGET_PER_CAT}")

    dup_secrets = [v for v, k in Counter(cred_values).items() if k > 1]
    if dup_secrets:
        errors.append(f"credential: {len(dup_secrets)} secret value(s) reused: {dup_secrets[:5]}")
    return errors


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{ln}: invalid JSON: {exc}")
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _report(rows: list[dict], errors: list[str]) -> None:
    counts = {c: sum(1 for r in rows if r.get("type") == c) for c in CATEGORIES}
    per = " / ".join(str(counts[c]) for c in CATEGORIES)
    print(f"{len(rows)} rows | {per} (plain/pii/credential/structured)")
    if errors:
        print(f"FAILED | {len(errors)} problem(s):")
        for e in errors[:40]:
            print(f"  - {e}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
    else:
        print("0 schema errors | 0 substring mismatches | 0 fake-data lint failures")


def _frame_report(tagged) -> None:
    """Print per-category frame diversity and credential-value uniqueness."""
    print("frame diversity (max repeat must be <= 3):")
    for cat in CATEGORIES:
        fc = Counter(fid for row, fid in tagged if row["type"] == cat)
        print(f"  {cat:11s}: {len(fc):>2} distinct frames, max repeat {max(fc.values())}")
    cred_vals = [row["key_entities"][1] for row, _ in tagged
                 if row["type"] == "credential" and len(row["key_entities"]) >= 2]
    print(f"credential secrets: {len(cred_vals)} values, "
          f"{len(set(cred_vals))} unique ({'OK' if len(cred_vals) == len(set(cred_vals)) else 'DUPES'})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build/validate the LeakLens ground-truth corpus.")
    default_out = Path(__file__).resolve().parent / "corpus.jsonl"
    ap.add_argument("--out", type=Path, default=default_out, help="output jsonl path")
    ap.add_argument("--seed", type=int, default=42, help="deterministic generation seed")
    ap.add_argument("--dry-run", action="store_true", help="build + validate + report, write nothing")
    ap.add_argument("--validate", type=Path, metavar="FILE",
                    help="validate an existing jsonl instead of building")
    args = ap.parse_args(argv)

    if args.validate is not None:
        rows = _load_jsonl(args.validate)
        errors = validate(rows)
        _report(rows, errors)
        return 1 if errors else 0

    tagged = _build_tagged(seed=args.seed)
    rows = [row for row, _ in tagged]
    errors = validate(rows)
    _report(rows, errors)
    _frame_report(tagged)
    if errors:
        return 1
    if args.dry_run:
        print("dry run - no file written")
        return 0
    _write_jsonl(rows, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
