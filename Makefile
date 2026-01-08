test:
	@uv run pytest

gui:
	@uv run python -m xsrp.gui

version:
	@grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'

release:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required. Usage: make release VERSION=0.1.1"; \
		exit 1; \
	fi
	@echo "Releasing version $(VERSION)..."
	@python3 -c "import re; f=open('pyproject.toml','r'); c=f.read(); f.close(); f=open('pyproject.toml','w'); f.write(re.sub(r'^version = \".*\"', 'version = \"$(VERSION)\"', c, flags=re.M)); f.close()"
	@git add pyproject.toml
	@git commit -m "Bump version to $(VERSION)" || true
	@git tag v$(VERSION)
	@echo "Created tag v$(VERSION)"
	@echo "Pushing tag to trigger PyPI release..."
	@git push origin v$(VERSION)
	@git push origin HEAD
	@echo "Release v$(VERSION) pushed! Check GitHub Actions for deployment status."

help:
	@echo "Available commands:"
	@echo "  make test              - Run tests"
	@echo "  make gui               - Run GUI application"
	@echo "  make version          - Show current version"
	@echo "  make release VERSION=X - Create new release (e.g., make release VERSION=0.1.1)"
	@echo "  make help             - Show this help message"