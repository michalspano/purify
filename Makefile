PREFIX ?= $(HOME)/.local

digitalize: go.mod $(wildcard *.go)
	CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o $@ .

install: digitalize
	install -d $(PREFIX)/bin && install -m 755 digitalize $(PREFIX)/bin/

clean:
	rm -f digitalize

.PHONY: install clean
