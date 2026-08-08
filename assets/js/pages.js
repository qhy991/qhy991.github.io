(() => {
    const buttons = [...document.querySelectorAll('[data-filter]')];
    const cards = [...document.querySelectorAll('[data-project-card]')];
    const search = document.querySelector('[data-project-search]');
    const results = document.querySelector('[data-results]');
    let activeFilter = 'all';

    if (!cards.length) return;

    const update = () => {
        const query = search?.value.trim().toLowerCase() || '';
        let visible = 0;

        cards.forEach((card) => {
            const tags = card.dataset.tags?.split(' ') || [];
            const matchesFilter = activeFilter === 'all' || tags.includes(activeFilter);
            const matchesQuery = !query || card.textContent.toLowerCase().includes(query);
            card.hidden = !(matchesFilter && matchesQuery);
            if (!card.hidden) visible += 1;
        });

        if (results) results.textContent = `${visible} project${visible === 1 ? '' : 's'} shown`;
    };

    buttons.forEach((button) => {
        button.addEventListener('click', () => {
            activeFilter = button.dataset.filter || 'all';
            buttons.forEach((item) => {
                const active = item === button;
                item.classList.toggle('is-active', active);
                item.setAttribute('aria-pressed', String(active));
            });
            update();
        });
    });

    search?.addEventListener('input', update);
    update();
})();
